#!/bin/bash
# MaleCNS Stage 2 런처: 다운로드(서버에서 직접) → 전처리 → 학습(파리 전체 CNS vs 무작위)
set -x
apt-get update -qq && apt-get install -qq -y python3.11 python3.11-venv > /dev/null 2>&1
python3.11 -m venv /root/venv 2>/dev/null || python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q numpy pandas pyarrow
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())" || { echo SETUP-FAIL; exit 1; }
echo SETUP-DONE

mkdir -p /root/work && cd /root/work
GCS="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
for i in $(seq 1 20); do
  sz=$(stat -c%s connectome-weights.feather 2>/dev/null || echo 0)
  [ "$sz" -ge 1051241946 ] && break
  curl -sL -C - -o connectome-weights.feather "$GCS/connectome-weights-male-cns-v1.0-minconf-0.5.feather" --max-time 500 || true
done
curl -sL -C - -o body-annotations.feather "$GCS/body-annotations-male-cns-v1.0-minconf-0.5.feather"
curl -sL -C - -o body-neurotransmitters.feather "$GCS/body-neurotransmitters-male-cns-v1.0.feather"
ls -la
echo DL-DONE

cat > train_stage2.py <<'PYEOF'
import json, time, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import urllib.request

torch.manual_seed(0); np.random.seed(0)
device = "cuda"
assert torch.cuda.is_available()

# ---------- 전처리 ----------
cw = pd.read_feather("connectome-weights.feather")
print("connectome:", cw.shape, cw.columns.tolist(), flush=True)
ann = pd.read_feather("body-annotations.feather")
print("annotations:", ann.shape, flush=True)
nt = pd.read_feather("body-neurotransmitters.feather")
print("nt:", nt.shape, flush=True)

# bodyId → index
ids = ann["bodyId"].to_numpy()
id2i = {b: i for i, b in enumerate(ids)}
src_g = cw["body_pre"].map(id2i)
tgt_g = cw["body_post"].map(id2i)
keep = src_g.notna() & tgt_g.notna()
src = src_g[keep].to_numpy().astype(np.int64)
tgt = tgt_g[keep].to_numpy().astype(np.int64)
syn = cw.loc[keep, "weight"].to_numpy().astype(np.float32)
n = len(ids)
print(f"graph n={n:,} E={len(src):,} syn_sum={syn.sum():,.0f}", flush=True)

# NT 기반 E/I 부호 (pre-synaptic 뉴런의 NT)
nt_col = None
for c in ["nt", "neurotransmitter", "top_nt", "predicted_nt"]:
    if c in nt.columns: nt_col = c; break
if nt_col is None:
    print("NT columns:", nt.columns.tolist(), flush=True)
nt_map = dict(zip(nt["bodyId"].to_numpy(), nt[nt_col].to_numpy()))
src_nt = np.array([nt_map.get(b, "unknown") for b in cw["body_pre"][keep].to_numpy()])
INH = {"gaba", "gaba,binding"}  # 억제성
sgn = np.where(np.isin(np.char.lower(src_nt.astype(str)), list(INH)), -1.0, 1.0).astype(np.float32)
print(f"E/I: +{int((sgn>0).sum()):,} / -{int((sgn<0).sum()):,}", flush=True)

# 셀타입
ct = ann["cellType"].astype(str).fillna("unk").to_numpy() if "cellType" in ann.columns else ann["type"].astype(str).fillna("unk").to_numpy()
uniq_t, type_of_node = np.unique(ct, return_inverse=True)
n_types = len(uniq_t)
print(f"cell types: {n_types}", flush=True)

# 입력(감각)/출력(운동) — annotations에서 역할 탐색
role_col = None
for c in ["superclass", "class", "cellClass", "flow"]:
    if c in ann.columns: role_col = c; break
print("role column:", role_col, flush=True)
roles = ann[role_col].astype(str).str.lower().to_numpy() if role_col else np.array(["x"]*n)
sens_mask = np.char.find(roles.astype(str), "sens") >= 0
out_mask = np.char.find(roles.astype(str), "motor") >= 0
input_idx = np.nonzero(sens_mask)[0]
output_idx = np.nonzero(out_mask)[0]
print(f"sensory={len(input_idx):,} motor={len(output_idx):,}", flush=True)
if len(input_idx) < 64 or len(output_idx) < 64:
    # 폴백: 무작위 슬롯
    g = np.random.default_rng(1)
    input_idx = g.permutation(n)[:512]
    output_idx = g.permutation(n)[512:1024]
    print("fallback IO slots used", flush=True)

def densify(st, tt):
    ps = st * n_types + tt
    up = np.unique(ps)
    rm = np.full(int(up.max())+1, -1, dtype=np.int64)
    rm[up] = np.arange(len(up))
    return rm[ps], len(up)

pair, n_pairs = densify(type_of_node[src], type_of_node[tgt])
print(f"pairs={n_pairs:,}", flush=True)
np.savez_compressed("malecns_graph.npz", n=n, src=src, tgt=tgt, syn=syn, sgn=sgn,
                    type_of_node=type_of_node, n_types=n_types,
                    input_idx=input_idx, output_idx=output_idx)
del cw; del ann; del nt

# ---------- 모델 (Stage 1과 동일 구조, IO 512) ----------
class SparseFlyRNN(nn.Module):
    def __init__(self, n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                 input_idx, output_idx, vocab, n_inj=256, n_out=512, seed=42):
        super().__init__()
        self.n = n
        self.register_buffer("indices", torch.tensor(np.stack([src, tgt]), dtype=torch.int64))
        self.register_buffer("pair", torch.from_numpy(pair))
        self.log_alpha = nn.Parameter(torch.zeros(n_pairs))
        self.type_bias = nn.Parameter(torch.zeros(n_types))
        self.tau = 0.5
        w0 = torch.log1p(torch.from_numpy(syn))
        self.register_buffer("w0", w0 / w0.max())
        self.register_buffer("sgn", torch.from_numpy(sgn))
        self.register_buffer("node_type", torch.from_numpy(type_of_node.astype(np.int64)))
        g = torch.Generator().manual_seed(seed)
        ii = torch.from_numpy(input_idx.astype(np.int64))
        oo = torch.from_numpy(output_idx.astype(np.int64))
        self.register_buffer("sensory", ii[torch.randperm(len(ii), generator=g)[:n_inj]])
        self.register_buffer("motor", oo[torch.randperm(len(oo), generator=g)[:n_out]])
        self.embed = nn.Parameter(torch.randn(vocab, n_inj) * 0.3)
        self.readout = nn.Parameter(torch.randn(n_out, vocab) * 0.1)

    def forward(self, h, tok):
        inj = self.embed[tok]
        h = h.index_copy(1, self.sensory, torch.tanh(h[:, self.sensory] + inj))
        alpha = torch.clamp(self.log_alpha.exp(), 0, 3)
        vals = self.w0 * self.sgn * alpha[self.pair]
        W = torch.sparse_coo_tensor(self.indices, vals, (self.n, self.n)).coalesce()
        acc = torch.sparse.mm(W, h.T).T
        bias = self.type_bias[self.node_type].unsqueeze(0)
        h = (1 - self.tau) * h + self.tau * torch.tanh(acc + bias)
        return torch.clamp(h, -10, 10)

    def logits(self, h):
        return h[:, self.motor] @ self.readout

if not os.path.exists("TinyStories-train.txt"):
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "TinyStories-train.txt")
text = open("TinyStories-train.txt", encoding="utf-8", errors="ignore").read(60_000_000)
CHARS = sorted(set(text)); V = len(CHARS)
c2i = {ch: i for i, ch in enumerate(CHARS)}
tokens = np.array([c2i[ch] for ch in text], dtype=np.int64)
del text
cnt = Counter(tokens.tolist())
p = np.array([cnt[i] for i in range(V)], dtype=np.float64); p /= p.sum()
UNIGRAM = float(-(p*np.log(p)).sum())
print(f"corpus {len(tokens):,} vocab {V} unigram {UNIGRAM:.4f}", flush=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train(model, tag, steps=8000, B=8, WARM=64, TBPTT=96, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    window = len(tokens) - WARM - TBPTT - 1
    rng = np.random.default_rng(0)
    losses = []
    t0 = time.time()
    for step in range(steps):
        base = int((step * 7 * B) % window)
        idx = (base + np.arange(B) * 9973 + rng.integers(0, 128, size=B)) % window
        batch = torch.tensor(np.stack([tokens[i:i+WARM+TBPTT+1] for i in idx]), device=device)
        h = torch.zeros(B, model.n, device=device)
        with torch.no_grad():
            for t in range(WARM):
                h = model(h, batch[:, t])
        loss = 0.0
        for t in range(WARM, WARM + TBPTT):
            h = model(h, batch[:, t])
            loss = loss + F.cross_entropy(model.logits(h), batch[:, t+1])
        loss = loss / TBPTT
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 200 == 0 or step == steps-1:
            mem = torch.cuda.max_memory_allocated()/1e9
            print(f"[{tag}] step {step:5d} loss {loss.item():.4f} mem {mem:.1f}GB [{time.time()-t0:.0f}s]", flush=True)
    return losses

fly = SparseFlyRNN(n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                   input_idx, output_idx, V)
print(f"FLY params {sum(p_.numel() for p_ in fly.parameters()):,}", flush=True)
losses_fly = train(fly, "FLY")
torch.save(fly.state_dict(), "malecns_fly.pt")
del fly; torch.cuda.empty_cache()

rng = np.random.default_rng(7)
src_r = rng.permutation(src); tgt_r = rng.permutation(tgt)
m = src_r != tgt_r
src_r, tgt_r = src_r[m], tgt_r[m]
type_r = rng.permutation(type_of_node).copy()
pair_r, n_pairs_r = densify(type_r[src_r], type_r[tgt_r])
rand = SparseFlyRNN(n, src_r, tgt_r, syn[:len(src_r)], sgn[:len(src_r)], pair_r, n_pairs_r,
                    type_r, n_types, input_idx, output_idx, V)
print(f"RANDOM E={len(src_r):,} pairs={n_pairs_r:,}", flush=True)
losses_rand = train(rand, "RANDOM")
torch.save(rand.state_dict(), "malecns_random.pt")

# 공정 평가 (동일 fresh 배치 40개, seed 123)
def evaluate(model, tag):
    model.eval()
    rng = np.random.default_rng(123)
    window = len(tokens) - WARM - TBPTT - 1
    tot = 0.0
    with torch.no_grad():
        for k in range(40):
            idx = rng.integers(0, window, size=B)
            batch = torch.tensor(np.stack([tokens[i:i+WARM+TBPTT+1] for i in idx]), device=device)
            h = torch.zeros(B, model.n, device=device)
            for t in range(WARM):
                h = model(h, batch[:, t])
            for t in range(WARM, WARM+TBPTT):
                h = model(h, batch[:, t])
                tot += F.cross_entropy(model.logits(h), batch[:, t+1]).item()
    model.train()
    return tot / (40 * TBPTT)

ev_fly = evaluate(fly, "FLY") if False else None  # fly는 del했으므로 재로드
fly2 = SparseFlyRNN(n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                    input_idx, output_idx, V)
fly2.load_state_dict(torch.load("malecns_fly.pt"))
ev_fly = evaluate(fly2, "FLY")
ev_rand = evaluate(rand, "RANDOM")

i2c = {i: c for c, i in c2i.items()}
@torch.no_grad()
def generate(model, prompt="Once upon a time ", n_chars=300, temperature=0.8):
    model.eval()
    h = torch.zeros(1, model.n, device=device)
    for ch in prompt:
        h = model(h, torch.tensor([c2i.get(ch, 0)], device=device))
    out = list(prompt)
    for _ in range(n_chars):
        lg = model.logits(h) / temperature
        nxt = torch.multinomial(F.softmax(lg, -1).squeeze(0), 1)
        out.append(i2c[int(nxt)])
        h = model(h, nxt)
    model.train()
    return "".join(out)

fly_txt = generate(fly2); rand_txt = generate(rand)
print("\n=== FLY sample ===\n" + fly_txt, flush=True)
print("\n=== RANDOM sample ===\n" + rand_txt, flush=True)

print(f"\n===== FINAL =====\nFLY {ev_fly:.4f}  RANDOM {ev_rand:.4f}  unigram {UNIGRAM:.4f}", flush=True)
print(f"FLY-RANDOM = {ev_fly-ev_rand:+.4f} ({'FLY WINS' if ev_fly < ev_rand else 'RANDOM >= FLY'})", flush=True)
json.dump({"fly": ev_fly, "random": ev_rand, "unigram": UNIGRAM,
           "fly_sample": fly_txt, "random_sample": rand_txt,
           "losses_fly": losses_fly[-100:], "losses_rand": losses_rand[-100:]},
          open("stage2_results.json", "w"))
print("ALL-DONE", flush=True)
PYEOF

nohup /root/venv/bin/python train_stage2.py > train2.log 2>&1 &
echo LAUNCHED
