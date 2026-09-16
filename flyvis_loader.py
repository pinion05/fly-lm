"""flyvis 내장 커넥톤(45,669뉴런) 로더 — flyvis API에서 직접 추출.

실측 (2026-09-14, flyvis 1.2.0):
  n_nodes=45,669 / n_edges=1,513,231 / cell types=65
  role: input 5,768 (R1-R8 광수용체) / intermediate 15,387 / output 24,514
  sign: +1 983,980 / -1 529,251 (실측 E/I 부호 — 유충과 달리 있음)
  n_syn: min 0.045 ~ max 144.09, sum 5,440,440
"""
from __future__ import annotations

import numpy as np
import torch

_EXTENT = 15  # flyvis 기본 배열 반경 (컬럼) → 45,669뉴런


def load_flyvis_builtin(extent: int = _EXTENT) -> dict:
    from flyvis.connectome import ConnectomeFromAvgFilters

    conn = ConnectomeFromAvgFilters(file="fib25-fib19_v2.2.json", extent=extent)
    nodes, edges = conn.nodes, conn.edges

    n = len(nodes.index[:])
    src = np.asarray(edges.source_index[:], dtype=np.int64)
    tgt = np.asarray(edges.target_index[:], dtype=np.int64)
    syn = np.asarray(edges.n_syn[:], dtype=np.float32)
    sign = np.asarray(edges.sign[:], dtype=np.float32)

    node_type = np.asarray(nodes.type[:])
    role = np.asarray(nodes.role[:])
    input_idx = np.nonzero(role == b"input")[0]
    output_idx = np.nonzero(role == b"output")[0]

    return {
        "n": n,
        "edge_index": torch.tensor(np.stack([src, tgt]), dtype=torch.int64),
        "syn_count": torch.tensor(syn, dtype=torch.float32),
        "signs": torch.tensor(sign, dtype=torch.float32),  # 진짜 E/I 부호 있음
        "node_type": node_type,          # 65종 셀타입 (weight sharing에 사용)
        "input_idx": input_idx,          # 5,768 감각류 → 문자 입력 슬롯
        "output_idx": output_idx,        # 24,514 하강/운동류 → readout 후보
    }


if __name__ == "__main__":
    c = load_flyvis_builtin()
    e = c["edge_index"]
    dens = e.shape[1] / (c["n"] * (c["n"] - 1))
    print(f"flyvis: n={c['n']:,} edges={e.shape[1]:,} density={dens*100:.4f}%")
    print(f"  cell_types={len(np.unique(c['node_type']))} input={len(c['input_idx']):,} output={len(c['output_idx']):,}")
    print(f"  sign: +{int((c['signs']>0).sum()):,} / -{int((c['signs']<0).sum()):,}  syn_sum={c['syn_count'].sum():,.0f}")
