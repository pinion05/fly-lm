"""Stage 0 스모크: 유충 커넥톤(2,952뉴런) 고정 배선 + 문자 단위 다음글자 예측.

설계 (flyvis 방식 이식, 단순화):
- 배선: 유충 all-all 연결에서 그대로 가져와 고정. 학습 대상 아님.
- 세기: "source 뉴런이 속한 세포종류 × target 뉴런이 속한 세포종류" 쌍마다 스칼라 1개
  (셀타입 어노테이션이 1,374행뿐이라 전부 못 덮으면 → 폴백: 전체 공유 스칼라 1개)
- 부호: [미확인] 유충 E/I 데이터 없음 → Stage 0은 전부 양수(+)로 시작, 학습 중 음수 방지 clamp
- 입력: 문자 원핫 → 무작위로 뽑은 64개 '감각 뉴런'에 프로젝션(이 프로젝션만 학습)
- 출력: 매 스텝 전체 2,952 뉴런 상태 → 64개 '운동 뉴런'에서 문자 어휘 로짓(readout만 학습)
- 상태 업데이트: leaky tanh 뉴런 (h = (1-τ)h + τ·tanh(W·h_in + b)) — 단순·안정 중심
- 안정장치: gradient clipping 1.0, 세기 clamp(0, 3), hidden state clamp(-10, 10)
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_data import load_larval

torch.manual_seed(0)
np.random.seed(0)

DATA = "the little girl was very happy. the dog ran to the park. "
CORPUS = (DATA * 400) + "the end."
CHARS = sorted(set(CORPUS))
V = len(CHARS)
c2i = {c: i for i, c in enumerate(CHARS)}
tokens = np.array([c2i[c] for c in CORPUS], dtype=np.int64)

N_SENS = 64          # 입력(감각) 뉴런 슬롯
SEQ = 96             # BPTT 길이
BATCH = 16
STEPS = 300          # 스모크라 짧게
LR = 3e-3


class FlyRNN(nn.Module):
    def __init__(self, n: int, edge_index: torch.Tensor, syn: torch.Tensor, vocab: int):
        super().__init__()
        self.n = n
        src, tgt = edge_index[0], edge_index[1]
        self.register_buffer("src", src)
        self.register_buffer("tgt", tgt)
        order = torch.argsort(self.tgt)  # scatter_agg 준비: target별 그룹
        self.register_buffer("order", order)
        self.register_buffer("tgt_sorted", self.tgt[order])
        # 학습 파라미터: 전역 세기 스칼라 1개 (flyvis의 Wsyn 아이디어) + bias 뉴런별 1개
        self.log_wsyn = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(n))
        self.tau = 0.5
        # 연결 세기: log(1+syn) 정규화, 고정
        w0 = torch.log1p(syn[order])
        self.register_buffer("w0", w0 / w0.max())
        # 입출력 배선
        g = torch.Generator().manual_seed(42)
        self.register_buffer("sensory", torch.randperm(n, generator=g)[:N_SENS])
        self.register_buffer("motor", torch.randperm(n, generator=g)[N_SENS:2 * N_SENS])
        self.embed = nn.Parameter(torch.randn(vocab, N_SENS) * 0.3)   # 문자→감각 주입
        self.readout = nn.Parameter(torch.randn(N_SENS, vocab) * 0.1)  # 운동→문자 로짓

    def forward(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        """h: [B, n], tok: [B] → 새 h 반환"""
        inj = self.embed[tok]                      # [B, N_SENS]
        h = h.index_copy(1, self.sensory, torch.tanh(h[:, self.sensory] + inj))
        wsyn = torch.clamp(self.log_wsyn.exp(), 0, 3)
        msg = h.index_select(1, self.src)          # [B, E]
        msg = msg * self.w0 * wsyn
        acc = torch.zeros_like(h)
        acc.index_add_(1, self.tgt, msg)
        h = (1 - self.tau) * h + self.tau * torch.tanh(acc + self.bias)
        return torch.clamp(h, -10, 10)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h[:, self.motor] @ self.readout     # [B, vocab]


def main():
    t0 = time.time()
    c = load_larval()
    n, edge_index, syn = c["n"], c["edge_index"], c["syn_count"]
    model = FlyRNN(n, edge_index, syn, V)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"n={n:,} edges={edge_index.shape[1]:,} vocab={V} params={sum(p.numel() for p in model.parameters()):,}")

    losses = []
    for step in range(STEPS):
        idx = np.random.randint(0, len(tokens) - SEQ - 1, size=BATCH)
        batch = torch.tensor(np.stack([tokens[i:i + SEQ + 1] for i in idx]))
        h = torch.zeros(BATCH, n)
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
            print(f"step {step:4d}  loss {loss.item():.4f}  grad_norm {gn.item():.2f}  wsyn {model.log_wsyn.exp().item():.3f}")

    first, last = np.mean(losses[:20]), np.mean(losses[-20:])
    print(f"\nfirst20 {first:.4f} → last20 {last:.4f}  ({(1-first/last)*-1:+.1%} 변화)")
    # 기준선: unigram loss
    cnt = Counter(tokens.tolist())
    p = np.array([cnt[i] for i in range(V)], dtype=np.float64); p /= p.sum()
    uni = -(p * np.log(p)).sum()
    print(f"unigram baseline {uni:.4f}  → 모델 last20 {'이김' if last < uni else '짐'} ({last-uni:+.4f})")
    print(f"소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
