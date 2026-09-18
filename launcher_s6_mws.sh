#!/bin/bash
# Stage 6: 시드 반복 실험 — 기존 20k 프로토콜(h=0, warm32, TBPTT32, 무경계 슬라이싱) 그대로,
# seed만 변경. 시드는 파일명에서 자동 판별: graph 파일명에 seed 태그 사용
#   malecns_graph.npz.s1 / .s2 같은 별칭 대신: 업로드되는 그래프에 따라 TAG=FLY/MASLOV,
#   SEED는 /root/work/SEED 파일에 적어두거나 기본 1
set -x
apt-get update -qq && apt-get install -qq -y python3-pip > /dev/null 2>&1
pip3 install -q --timeout 180 --retries 10 numpy torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch ok')" 2>/dev/null || pip3 install -q --force-reinstall --no-cache-dir --timeout 180 --retries 10 torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch ok')" || { echo SETUP-FAIL; exit 1; }
echo SETUP-DONE

mkdir -p /root/work && cd /root/work

cat > train_s6.py <<'PYEOF'
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, time, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

TAG = sys.argv[1] if len(sys.argv) > 1 else "FLY"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
GRAPH = "malecns_graph.npz" if TAG == "FLY" else ("malecns_mws.npz" if TAG == "MWS" else "malecns_maslov.npz")

torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda"
assert torch.cuda.is_available()

if not os.path.exists("TinyStories-train.txt"):
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "TinyStories-train.txt")

d = np.load(GRAPH)
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

class SparseFlyRNN(nn.Module):
    def __init__(self, n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                 input_idx, output_idx, vocab, n_inj=256, n_out=512, seed=42):
        super().__init__()
        self.n = n
        self.register_buffer("src", torch.tensor(src, dtype=torch.int64))
        self.register_buffer("tgt", torch.tensor(tgt, dtype=torch.int64))
        self.register_buffer("pair", torch.from_numpy(pair))
        self.log_alpha = nn.Parameter(torch.zeros(n_pairs))
        self.type_bias = nn.Parameter(torch.zeros(n_types))
        self.tau = 0.5
        w0 = torch.log1p(torch.from_numpy(syn))
        self.register_buffer("w0", w0 / w0.max())
        self.register_buffer("sgn", torch.from_numpy(sgn))
        self.register_buffer("node_type", torch.from_numpy(type_of_node.astype(np.int64)))
        g = torch.Generator().manual_seed(seed + SEED)
        ii = torch.from_numpy(input_idx.astype(np.int64))
        oo = torch.from_numpy(output_idx.astype(np.int64))
        self.register_buffer("sensory", ii[torch.randperm(len(ii), generator=g)[:n_inj]])
        self.register_buffer("motor", oo[torch.randperm(len(oo), generator=g)[:n_out]])
        self.embed = nn.Parameter(torch.randn(vocab, n_inj, generator=g) * 0.3)
        self.readout = nn.Parameter(torch.randn(n_out, vocab, generator=g) * 0.1)
    def forward(self, h, tok):
        inj = self.embed[tok]
        h = h.index_copy(1, self.sensory, torch.tanh(h[:, self.sensory] + inj))
        alpha = torch.clamp(self.log_alpha.exp(), 0, 3)
        vals = self.w0 * self.sgn * alpha[self.pair]
        msg = h.index_select(1, self.src) * vals.unsqueeze(0)
        acc = torch.zeros_like(h)
        acc = acc.index_add(1, self.tgt, msg)
        bias = self.type_bias[self.node_type].unsqueeze(0)
        h = (1 - self.tau) * h + self.tau * torch.tanh(acc + bias)
        return torch.clamp(h, -10, 10)
    def logits(self, h):
        return h[:, self.motor] @ self.readout

text = open("TinyStories-train.txt", encoding="utf-8", errors="ignore").read(60_000_000)
CHARS = sorted(set(text)); V = len(CHARS)
c2i = {ch: i for i, ch in enumerate(CHARS)}
i2c = {i: c for c, i in c2i.items()}
tokens = np.array([c2i[ch] for ch in text], dtype=np.int64)
del text
cnt = Counter(tokens.tolist())
p = np.array([cnt[i] for i in range(V)], dtype=np.float64); p /= p.sum()
UNIGRAM = float(-(p*np.log(p)).sum())
print(f"[{TAG}/seed{SEED}] corpus {len(tokens):,} vocab {V} unigram {UNIGRAM:.4f}", flush=True)

model = SparseFlyRNN(n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                     input_idx, output_idx, V)
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

B, WARM, TBPTT = 2, 32, 32   # 기존 20k 프로토콜 완전 동일
TOTAL = 20000
losses = []
t0 = time.time()
rng = np.random.default_rng(SEED)  # 배치 샘플링도 시드 의존

for step in range(TOTAL):
    starts = rng.integers(0, len(tokens)-WARM-TBPTT-2, size=B)
    rows = []
    for s in starts:
        rows.append(tokens[s:s+WARM+TBPTT+1])
    batch = torch.tensor(np.stack(rows), device=device)
    h = torch.zeros(B, model.n, device=device)
    with torch.no_grad():
        for t in range(WARM):
            h = model(h, batch[:, t])
    h = h.detach()
    loss = 0.0
    for t in range(WARM, WARM+TBPTT):
        h = model(h, batch[:, t])
        loss = loss + F.cross_entropy(model.logits(h), batch[:, t+1])
    loss = loss / TBPTT
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 200 == 0 or step == TOTAL-1:
        mem = torch.cuda.max_memory_allocated()/1e9
        print(f"[{TAG}S6x{SEED}] step {step:5d} loss {loss.item():.4f} mem {mem:.1f}GB [{time.time()-t0:.0f}s]", flush=True)

# 공정평가 (기존과 동일 프로토콜: 랜덤 시작, warm 32, eval 32, 40배치)
# 주의: 평가 시드는 항상 123 (학습 시드와 독립) — 시드 간 비교 가능하게
model.eval()
rng_e = np.random.default_rng(123)
tot = 0.0
with torch.no_grad():
    for k in range(40):
        starts = rng_e.integers(0, len(tokens)-WARM-TBPTT-2, size=B)
        rows = []
        for s in starts:
            rows.append(tokens[s:s+WARM+TBPTT+1])
        batch = torch.tensor(np.stack(rows), device=device)
        h = torch.zeros(B, model.n, device=device)
        for t in range(WARM):
            h = model(h, batch[:, t])
        for t in range(WARM, WARM+TBPTT):
            h = model(h, batch[:, t])
            tot += F.cross_entropy(model.logits(h), batch[:, t+1]).item()
ev = tot / (40 * TBPTT)
print(f"\n===== FINAL [{TAG}/seed{SEED}] =====\nS6 eval {ev:.4f} (seed0 FLY 2.332 / MASLOV 2.569 / unigram {UNIGRAM:.4f})", flush=True)
json.dump({"s6": ev, "tag": TAG, "seed": SEED, "unigram": UNIGRAM, "losses": losses[::20]},
          open(f"s6_{TAG}_seed{SEED}_results.json", "w"))
print("ALL-DONE", flush=True)
PYEOF

mkdir -p /root/work
if [ -f /root/work/malecns_mws.npz ]; then
  TAG=MWS
elif [ -f /root/work/malecns_maslov.npz ]; then
  TAG=MASLOV
else
  TAG=FLY
fi
SEED=$(cat /root/work/SEED 2>/dev/null || echo 1)
nohup python3 train_s6.py $TAG $SEED > train_s6_${TAG}_s${SEED}.log 2>&1 &
echo LAUNCHED-TAG=$TAG-SEED=$SEED
