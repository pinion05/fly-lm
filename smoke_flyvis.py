"""flyvis 45,669뉴런 커넥톤 + TinyStories 문자단위 다음글자 예측 (Stage 0 본 검증).

유충 스모크(smoke_larval.py)와 동일 구조에 flyvis 실측 데이터 적용:
- E/I 부호: flyvis sign 배열 그대로 고정 (학습 안 함) — msg에 곱해줌
- 세기: 전역 wsyn 1개 + bias 뉴런별 (flyvis의 alpha는 셀타입별이지만
  Stage 0에서는 전역 1개로 시작 — 셀타입 공유는 다음 단계)
- 입력: 문자 → input 뉴런 5,768개 중 64개 슬롯 (flyvis input role 사용!)
- 출력: output 뉴런 24,514개 중 64개 슬롯에서 문자 로짓
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flyvis_loader import load_flyvis_builtin

torch.manual_seed(0)
np.random.seed(0)

DATA_PATH = Path.home() / ".cache/fly-lm/data/TinyStories-train.txt"
N_INJ = 64          # 입력 슬롯
N_OUT = 64          # 출력 슬롯
SEQ = 96
BATCH = 16
STEPS = 400
LR = 3e-3
CHUNK = 20_000_000  # 코퍼스에서 사용할 바이트 (26MB 파일이 아직 resume 중일 수 있음)


class FlyRNN(nn.Module):
    def __init__(self, n, edge_index, syn, signs, input_idx, output_idx, vocab):
        super().__init__()
        self.n = n
        self.register_buffer("src", edge_index[0])
        self.register_buffer("tgt", edge_index[1])
        order = torch.argsort(self.tgt)
        self.register_buffer("order", order)
        # 학습 파라미터: 전역 세기 + 뉴런별 bias (Stage 0 단순형)
        self.log_wsyn = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(n))
        self.tau = 0.5
        w0 = torch.log1p(syn[order])
        self.register_buffer("w0", w0 / w0.max())
        self.register_buffer("sgn", signs[order])          # E/I 부호 고정
        g = torch.Generator().manual_seed(42)
        # 입력은 input role 뉴런에서, 출력은 output role 뉴런에서만
        ii = torch.from_numpy(input_idx.astype(np.int64))
        oo = torch.from_numpy(output_idx.astype(np.int64))
        self.register_buffer("sensory", ii[torch.randperm(len(ii), generator=g)[:N_INJ]])
        self.register_buffer("motor", oo[torch.randperm(len(oo), generator=g)[:N_OUT]])
        self.embed = nn.Parameter(torch.randn(vocab, N_INJ) * 0.3)
        self.readout = nn.Parameter(torch.randn(N_OUT, vocab) * 0.1)

    def forward(self, h, tok):
        inj = self.embed[tok]
        h = h.index_copy(1, self.sensory, torch.tanh(h[:, self.sensory] + inj))
        wsyn = torch.clamp(self.log_wsyn.exp(), 0, 3)
        msg = h.index_select(1, self.src)
        msg = msg * self.w0 * self.sgn * wsyn             # 부호×세기, 부호는 고정
        acc = torch.zeros_like(h)
        acc.index_add_(1, self.tgt, msg)
        h = (1 - self.tau) * h + self.tau * torch.tanh(acc + self.bias)
        return torch.clamp(h, -10, 10)

    def logits(self, h):
        return h[:, self.motor] @ self.readout


def main():
    t0 = time.time()
    c = load_flyvis_builtin()
    text = open(DATA_PATH, encoding="utf-8", errors="ignore").read(CHUNK)
    CHARS = sorted(set(text))
    V = len(CHARS)
    c2i = {ch: i for i, ch in enumerate(CHARS)}
    tokens = np.array([c2i[ch] for ch in text], dtype=np.int64)
    print(f"corpus {len(tokens):,} chars, vocab {V}, n={c['n']:,}, E={c['edge_index'].shape[1]:,}")

    model = FlyRNN(c["n"], c["edge_index"], c["syn_count"], c["signs"],
                   c["input_idx"], c["output_idx"], V)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"trainable params: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    for step in range(STEPS):
        idx = np.random.randint(0, len(tokens) - SEQ - 1, size=BATCH)
        batch = torch.tensor(np.stack([tokens[i:i + SEQ + 1] for i in idx]))
        h = torch.zeros(BATCH, c["n"])
        loss = 0.0
        for t in range(SEQ):
            h = model(h, batch[:, t])
            loss = loss + F.cross_entropy(model.logits(h), batch[:, t + 1])
        loss = loss / SEQ
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  gn {gn.item():.2f}  wsyn {model.log_wsyn.exp().item():.3f}")

    first, last = np.mean(losses[:20]), np.mean(losses[-20:])
    cnt = Counter(tokens.tolist())
    p = np.array([cnt[i] for i in range(V)], dtype=np.float64); p /= p.sum()
    uni = -(p * np.log(p)).sum()
    print(f"\nfirst20 {first:.4f} → last20 {last:.4f}")
    print(f"unigram {uni:.4f} → 모델 {'이김' if last < uni else '짐'} ({last-uni:+.4f})")
    print(f"소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
