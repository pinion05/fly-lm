import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TAG = sys.argv[1] if len(sys.argv) > 1 else "FLY"
CKPT = sys.argv[2] if len(sys.argv) > 2 else f"s4_{TAG}_final.pt"
GRAPH = "malecns_graph.npz" if TAG == "FLY" else "malecns_maslov.npz"

torch.manual_seed(0); np.random.seed(0)
device = "cuda"
assert torch.cuda.is_available()

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

model = SparseFlyRNN(n, src, tgt, syn, sgn, pair, n_pairs, type_of_node, n_types,
                     input_idx, output_idx, V)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.to(device).eval()
print(f"[{TAG}] loaded {CKPT}", flush=True)

tv = tokens[:5_000_000]
alpha_mask = np.isin(tv, sorted(c2i[c] for c in CHARS if c.isalpha()))
word_starts = np.where(alpha_mask & ~np.roll(alpha_mask, 1))[0]
window_pool = word_starts[(word_starts > 32) & (word_starts < 5_000_000 - 500)]

# ===== 공정 eval: word-start 정렬, 64 warm + 64 eval (기존 20k 프로토콜: WARM=64, TBPTT=96→S4는 64로 축소, seed 123, nb=40) =====
WARM, EVLEN, B = 64, 64, 2
rng_e = np.random.default_rng(123)
tot, cnt = 0.0, 0
with torch.no_grad():
    for k in range(40):
        starts = rng_e.choice(window_pool, size=B)
        batch = torch.tensor(np.stack([tv[s:s+WARM+EVLEN+1] for s in starts]), device=device)
        h = torch.zeros(B, model.n, device=device)
        for t in range(WARM):
            h = model(h, batch[:, t])
        for t in range(WARM, WARM+EVLEN):
            h = model(h, batch[:, t])
            tot += F.cross_entropy(model.logits(h), batch[:, t+1]).item()
            cnt += 1
ev = tot / cnt
print(f"S4-EVAL [{TAG}] {ev:.4f} (기존 20k: FLY 2.332 / MASLOV 2.569 / unigram 3.080)", flush=True)

out = {"s4_eval": ev, "tag": TAG, "ckpt": CKPT, "protocol": {"warm": WARM, "evlen": EVLEN, "B": B, "nb": 40, "seed": 123, "align": "word-start"}}
json.dump(out, open(f"s4eval_{TAG}.json", "w"))
print("EVAL-DONE", flush=True)
