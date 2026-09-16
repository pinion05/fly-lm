"""셀타입별 세기 공유 모델 (flyvis식 alpha) — 배선 구조를 성능으로 번역하는 단계.

기존 FlyRNN과의 차이:
- 전역 wsyn 1개 → (source 셀타입 × target 셀타입) 쌍별 세기 alpha
  flyvis 커넥톤은 65 cell type → 관측되는 (src,tar) 조합만 파라미터화 (약 500~600개)
- bias는 셀타입별로 공유 (65개) — 뉴런별 45,669개에서 축소
- E/I 부호와 시냅스 수는 여전히 고정 (측정값)

무작위 배선 컨트롤에는 셀타입이 없음 → 같은 수의 alpha를 무작위 그룹에 배정해서
동일 자유도를 맞춘다 (공정 비교).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from flyvis_loader import load_flyvis_builtin


class CellTypeRNN(nn.Module):
    def __init__(self, edge_index, syn, signs, src_type, tgt_type, n_types,
                 node_type_idx, input_idx, output_idx, vocab, n_inj=64, n_out=64):
        super().__init__()
        E = edge_index.shape[1]
        self.register_buffer("src", edge_index[0])
        self.register_buffer("tgt", edge_index[1])
        # (src_type, tgt_type) 쌍을 관측 쌍만稠密 재번호 (flyvis의 604개)
        pair_sparse = src_type * n_types + tgt_type
        uniq_pairs = torch.unique(pair_sparse)
        remap = torch.full((int(uniq_pairs.max()) + 1,), -1, dtype=torch.int64)
        remap[uniq_pairs] = torch.arange(len(uniq_pairs))
        pair = remap[pair_sparse]
        self.n_pairs = int(len(uniq_pairs))
        self.register_buffer("pair", pair)
        # 학습 파라미터: 쌍별 세기(양수 클램프) + 셀타입별 bias
        self.log_alpha = nn.Parameter(torch.zeros(self.n_pairs))
        self.type_bias = nn.Parameter(torch.zeros(n_types))
        self.tau = 0.5
        w0 = torch.log1p(syn)
        self.register_buffer("w0", w0 / w0.max())
        self.register_buffer("sgn", signs)
        self.register_buffer("node_type_idx", node_type_idx)   # [n] 노드별 타입 → bias 색인
        g = torch.Generator().manual_seed(42)
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
        msg = h.index_select(1, self.src)
        msg = msg * self.w0 * self.sgn * alpha[self.pair]
        acc = torch.zeros_like(h)
        acc.index_add_(1, self.tgt, msg)
        bias = self.type_bias[self.node_type_idx]       # [n]
        h = (1 - self.tau) * h + self.tau * torch.tanh(acc + bias.unsqueeze(0))
        return torch.clamp(h, -10, 10)

    def logits(self, h):
        return h[:, self.motor] @ self.readout


def load_celltype_data():
    """flyvis 로드 + edge별 셀타입 인덱스 추출."""
    c = load_flyvis_builtin()
    import numpy as np
    node_type = c["node_type"]                    # bytes
    uniq, type_of_node = np.unique(node_type, return_inverse=True)
    n_types = len(uniq)
    src_t = type_of_node[c["edge_index"][0].numpy()]
    tgt_t = type_of_node[c["edge_index"][1].numpy()]
    c["src_type"] = src_t.astype(np.int64)
    c["tgt_type"] = tgt_t.astype(np.int64)
    c["n_types"] = n_types
    c["type_of_node"] = type_of_node
    return c


if __name__ == "__main__":
    c = load_celltype_data()
    pair = c["src_type"] * c["n_types"] + c["tgt_type"]
    print(f"n={c['n']:,} E={c['edge_index'].shape[1]:,} cell_types={c['n_types']}")
    print(f"관측 (src,tar) 쌍 수 = 실제 alpha 파라미터 수 = {len(np.unique(pair))}")
