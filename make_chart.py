#!/usr/bin/env python3
"""3-way 학습 곡선 정적 차트 — README용 (matplotlib)."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path.home()

# FLY/RANDOM 스텝별 loss — stage2 로그에서
log = (HOME / "fly-lm/stage2_train.log").read_text()
pts_fly, pts_rnd = [], []
for m in re.finditer(r"\[(FLY|RANDOM)\] step\s+(\d+) loss ([0-9.]+)", log):
    (pts_fly if m.group(1) == "FLY" else pts_rnd).append((int(m.group(2)), float(m.group(3))))

# MASLOV — train_maslov.log
mlog = (HOME / "fly-lm/train_maslov.log").read_text()
pts_msl = [(int(s), float(l)) for s, l in re.findall(r"\[MASLOV\] step\s+(\d+) loss ([0-9.]+)", mlog)]
# MASLOV는 maslov_results.json에 losses_tail만 있음 — 로그에서 스텝별 추출

UNIGRAM = 3.0802

def smooth(pts, w=5):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sm = []
    for i in range(len(ys)):
        lo = max(0, i - w // 2)
        hi = min(len(ys), i + w // 2 + 1)
        sm.append(sum(ys[lo:hi]) / (hi - lo))
    return xs, sm

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor("#0b0e14")
ax.set_facecolor("#10141c")

for pts, color, label in [
    (pts_fly, "#4cc38a", "Fly wiring (MaleCNS) — final eval 2.629"),
    (pts_msl, "#7aa2f7", "Degree-preserving rewired — final eval 2.952"),
    (pts_rnd, "#f5a97f", "Random wiring — final eval 3.112"),
]:
    if not pts:
        continue
    xs, sm = smooth(pts)
    ax.plot(xs, sm, color=color, linewidth=2.2, label=label)

ax.axhline(UNIGRAM, color="#8b93a7", linestyle="--", linewidth=1.2)
ax.text(150, UNIGRAM + 0.03, "unigram baseline 3.080", color="#8b93a7", fontsize=9)

ax.set_xlabel("training step", color="#e6e9ef")
ax.set_ylabel("loss (nats/char)", color="#e6e9ef")
ax.set_title("fly-lm: char-level LM on the full adult fly CNS — 3-way wiring control", color="#e6e9ef", fontsize=12)
ax.tick_params(colors="#8b93a7")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#3a4152")
ax.spines["bottom"].set_color("#3a4152")
leg = ax.legend(facecolor="#151a23", edgecolor="#3a4152", labelcolor="#e6e9ef", fontsize=9)
ax.set_ylim(2.3, 5.0)

fig.tight_layout()
out = HOME / "fly-lm/assets/training_curves.png"
fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved", out, "| points:", len(pts_fly), len(pts_msl), len(pts_rnd))
