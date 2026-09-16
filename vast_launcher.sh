#!/bin/bash
# fly-lm Stage 1 Vast.ai 런처: 세팅 → 학습(파리+무작위 4000+4000) → 결과 JSON
set -x
# 1) 파이썬 환경
apt-get update -qq && apt-get install -qq -y python3.11 python3.11-venv git > /dev/null 2>&1 || apt-get install -qq -y python3-pip > /dev/null 2>&1
python3.11 -m venv /root/venv 2>/dev/null || python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q numpy
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { echo SETUP-FAIL; exit 1; }
echo SETUP-DONE

# 2) 학습 스크립트 (base64로 올린 fly_connectome.npz + TinyStories 다운로드)
mkdir -p /root/work && cd /root/work

cat > train_stage1.py <<'PYEOF'
import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import urllib.request, os

torch.manual_seed(0); np.random.seed(0)
device = "cuda"
assert torch.cuda.is_available(), "GPU 필요"

if not os.path.exists("/root/work/TinyStories-train.txt"):
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "/root/work/TinyStories-train.txt")
print(f"corpus {os.path.getsize('/root/work/TinyStories-train.txt')/1e6:.0f}MB", flush=True)

d = np.load("/root/work/fly_connectome.npz")
n = int(d["n"]); src, tgt = d["src"], d["tgt"]
syn, sgn = d["syn"], d["sgn"]
type_of_node = d["type_of_node"]; n_types = int(d["n_types"])
input_idx, output_idx = d["input_idx"], d["output_idx"]

def densify(st, tt):
    ps = st * n_types + tt
    up = np.unique(ps)
    rm = np.full(int(up.max())+1, -1, dtype=np.int64)
    rm[up] = np.arange(len(up))
    return rm[ps], len(up)

pair, n_pairs = densify(type_of_node[src], type_of_node[tgt])
print(f"connectome n={n:,} E={len(src):,} pairs={n_pairs}", flush=True)

class SparseFlyRNN(nn.Module):
    def __init__(self, n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                 input_idx, output_idx, vocab, n_inj=128, n_out=128, seed=42):
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

text = open("/root/work/TinyStories-train.txt", encoding="utf-8", errors="ignore").read(60_000_000)
CHARS = sorted(set(text)); V = len(CHARS)
c2i = {ch: i for i, ch in enumerate(CHARS)}
tokens = np.array([c2i[ch] for ch in text], dtype=np.int64)
del text
cnt = Counter(tokens.tolist())
p = np.array([cnt[i] for i in range(V)], dtype=np.float64); p /= p.sum()
UNIGRAM = float(-(p*np.log(p)).sum())
print(f"corpus {len(tokens):,} vocab {V} unigram {UNIGRAM:.4f}", flush=True)

def train(model, tag, steps=4000, B=16, WARM=64, TBPTT=96, lr=1e-3):
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
        if step % 100 == 0 or step == steps-1:
            mem = torch.cuda.max_memory_allocated()/1e9
            print(f"[{tag}] step {step:5d} loss {loss.item():.4f} mem {mem:.1f}GB [{time.time()-t0:.0f}s]", flush=True)
    return losses

fly = SparseFlyRNN(n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                   input_idx, output_idx, V)
print(f"FLY params {sum(p_.numel() for p_ in fly.parameters()):,}", flush=True)
losses_fly = train(fly, "FLY")
torch.save(fly.state_dict(), "/root/work/celltype_fly.pt")

rng = np.random.default_rng(7)
src_r = rng.permutation(src); tgt_r = rng.permutation(tgt)
m = src_r != tgt_r
src_r, tgt_r = src_r[m], tgt_r[m]
type_r = rng.permutation(type_of_node).copy()
pair_r, n_pairs_r = densify(type_r[src_r], type_r[tgt_r])
rand = SparseFlyRNN(n, src_r, tgt_r, syn[:len(src_r)], sgn[:len(src_r)], pair_r, n_pairs_r,
                    type_r, n_types, input_idx, output_idx, V)
print(f"RANDOM E={len(src_r):,} pairs={n_pairs_r}", flush=True)
losses_rand = train(rand, "RANDOM")
torch.save(rand.state_dict(), "/root/work/celltype_random.pt")

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
        nxt = torch.multinomial(F.softmax(lg, -1), 1)
        out.append(i2c[int(nxt)])
        h = model(h, nxt)
    model.train()
    return "".join(out)

fly_txt = generate(fly); rand_txt = generate(rand)
print("\n=== FLY sample ===\n" + fly_txt, flush=True)
print("\n=== RANDOM sample ===\n" + rand_txt, flush=True)

lf, lr_ = np.mean(losses_fly[-50:]), np.mean(losses_rand[-50:])
print(f"\n===== FINAL =====\nFLY {lf:.4f}  RANDOM {lr_:.4f}  unigram {UNIGRAM:.4f}", flush=True)
print(f"FLY-RANDOM = {lf-lr_:+.4f} ({'FLY WINS' if lf < lr_ else 'RANDOM >= FLY'})", flush=True)
json.dump({"fly": losses_fly, "random": losses_rand, "unigram": UNIGRAM,
           "fly_sample": fly_txt, "random_sample": rand_txt},
          open("/root/work/stage1_results.json", "w"))
print("ALL-DONE", flush=True)
PYEOF

nohup python train_stage1.py > /root/work/train.log 2>&1 &
echo LAUNCHED
