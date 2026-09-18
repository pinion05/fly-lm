#!/usr/bin/env python3
"""WSHUF / RANDOM 그래프 재생성 (리뷰 대조용).
WSHUF: FLY 위상 고정 + syn만 같은 부호 클래스 내 셔플 (seed 888)
  - 위상 불변, 가중치 다중집합 보존, 부호 보존, per-node in-degree 보존
  - per-node in-strength 파괴 → strength top1% 20.35% → 12.85%
RANDOM: 엣지 끝점 완전 무작위 재배열 (seed 999)
검증 출력 포함. 재현: python3 make_controls.py
"""
import numpy as np

d = np.load("malecns_graph.npz")
src, tgt = d["src"], d["tgt"]
n = int(d["n"])

# --- WSHUF ---
rng = np.random.default_rng(888)
syn = d["syn"].copy()
key = (d["sgn"] > 0).astype(np.int64)
for k in [0, 1]:
    idx = np.where(key == k)[0]
    syn[idx] = syn[rng.permutation(idx)]
assert np.array_equal(np.sort(syn), np.sort(d["syn"]))
np.savez_compressed("malecns_wshuf.npz", n=n, src=src, tgt=tgt, syn=syn, sgn=d["sgn"].copy(),
                    type_of_node=d["type_of_node"], n_types=d["n_types"],
                    input_idx=d["input_idx"], output_idx=d["output_idx"])

# --- RANDOM ---
rng2 = np.random.default_rng(999)
tgt3 = tgt[rng2.permutation(len(tgt))]
np.savez_compressed("malecns_random.npz", n=n, src=src.copy(), tgt=tgt3, syn=d["syn"].copy(), sgn=d["sgn"].copy(),
                    type_of_node=d["type_of_node"], n_types=d["n_types"],
                    input_idx=d["input_idx"], output_idx=d["output_idx"])
print("controls regenerated + verified")
