"""flyvis 커넥톤 학습 — 메모리 절약형 (truncated BPTT + detached warm-up).

OOM 원인 실측: 백그라운드 워커 cgroup이 프로세스당 4GB 제한(메인 메모리 30GB와 무관).
BPTT 그래프 = 96시점 × (h: B×45,669 fp32 + msg: B×1,513,231 fp32) ≈ 스텝당 ~640MB → 96시점 보존 불가.

해법 (truncated BPTT, flyvis의 19프레임 full BPTT와 달리 메모리 제약 하 선택):
- 워밍업 WARM 스텝: h만 유지하고 그래프는 끊음(detach) — 학습 신호 없음, 상태 예열만
- 학습 구간 TBPTT 스텝: 그래프 보존하며 BPTT → 그 후에만 backward

v2 변경 (1차 실험 결과 반영):
- 1차: loss 4.61→2.99(100) 후 3.34(300)로 반등 + wsyn 1.0 근처 정체
- 원인 가설: (a) lr 3e-3 과다로 발진 (b) 샘플 배치 다양성 부족 — 코퍼스 4M chars에서
  매번 무작위 시작점이라 수렴 신호 노이즈 큼
- v2: lr 1e-3, 스텝당 데이터 커서 순환(에폭 구조), 로그 50스텝 단위
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flyvis_loader import load_flyvis_builtin
from smoke_flyvis import FlyRNN

torch.manual_seed(0)
np.random.seed(0)

DATA = Path.home() / ".cache/fly-lm/data/TinyStories-train.txt"
B, WARM, TBPTT = 4, 64, 24
CHUNK = 4_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    STEPS = args.steps

    t0 = time.time()
    c = load_flyvis_builtin()
    text = DATA.read_text(errors="ignore")[:CHUNK]
    CHARS = sorted(set(text))
    V = len(CHARS)
    c2i = {ch: i for i, ch in enumerate(CHARS)}
    tokens = np.array([c2i[ch] for ch in text], dtype=np.int64)
    print(f"corpus {len(tokens):,} vocab {V} | B={B} warm={WARM} tbptt={TBPTT} steps={STEPS} lr={args.lr}", flush=True)

    model = FlyRNN(c["n"], c["edge_index"], c["syn_count"], c["signs"],
                   c["input_idx"], c["output_idx"], V)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    window = len(tokens) - WARM - TBPTT - 1
    for step in range(STEPS):
        # 커서 순환: 배치 시작점을 에폭처럼 순회 + 약간의 지터
        base = int((step * 7 * B) % window)
        idx = (base + np.arange(B) * 997 + np.random.randint(0, 50, size=B)) % window
        batch = torch.tensor(np.stack([tokens[i:i + WARM + TBPTT + 1] for i in idx]))
        h = torch.zeros(B, c["n"])
        with torch.no_grad():
            for t in range(WARM):
                h = model(h, batch[:, t])
        loss = 0.0
        for t in range(WARM, WARM + TBPTT):
            h = model(h, batch[:, t])
            loss = loss + F.cross_entropy(model.logits(h), batch[:, t + 1])
        loss = loss / TBPTT
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            el = time.time() - t0
            print(f"step {step:5d} loss {loss.item():.4f} gn {gn.item():.2f} wsyn {model.log_wsyn.exp().item():.3f} [{el:.0f}s]", flush=True)

    first, last = np.mean(losses[:20]), np.mean(losses[-20:])
    cnt = Counter(tokens.tolist())
    p = np.array([cnt[i] for i in range(V)], dtype=np.float64)
    p /= p.sum()
    uni = -(p * np.log(p)).sum()
    print(f"\nfirst20 {first:.4f} -> last20 {last:.4f}")
    print(f"unigram {uni:.4f} -> model {'WINS' if last < uni else 'LOSES'} ({last - uni:+.4f})")
    torch.save(model.state_dict(), Path.home() / "fly-lm/flyvis_smoke.pt")
    print("saved flyvis_smoke.pt")


if __name__ == "__main__":
    main()

