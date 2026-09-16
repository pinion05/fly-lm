"""파리 커넥톤 로더: 유충(3,016)과 flyvis 내장 커넥톤(45,669)을 하나의 형식으로.

반환 형식 (두 소스 공통):
  n: 뉴런 수
  edge_index: [2, E] int64 — 방향 있는 연결 (source, target)
  syn_count: [E] float32 — 시냅스 수 (연결 세기의 대리값)
  signs: [+1/-1] 소스별로 주어지면 사용, 없으면 [추론] 전부 +1로 시작(Stage 0 한정)
"""
from pathlib import Path
import json
import zipfile
import urllib.request

import numpy as np
import torch

LARVAL_URL = "https://github.com/brain-networks/larval-drosophila-connectome/raw/main/Supplementary-Data-S1.zip"
LARVAL_CACHE = Path.home() / ".cache" / "fly-lm" / "larval"


def load_larval() -> dict:
    """Winding et al. 2023 유충 커넥톤 (3,016뉴런). 1.11MB zip → npz."""
    cache = LARVAL_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    npz_path = cache / "larval.npz"
    if npz_path.exists():
        with np.load(npz_path) as d:
            adj = d["adj"]
    else:
        zip_path = cache / "S1.zip"
        if not zip_path.exists():
            print("다운로드:", LARVAL_URL)
            urllib.request.urlretrieve(LARVAL_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            # 인접행렬은 CSV: ad(axon→dendrite)가 방향성 표준
            with z.open("Supplementary-Data-S1/ad_connectivity_matrix.csv") as f:
                import csv as _csv
                rows = list(_csv.reader(f.read().decode("utf-8").splitlines()))
            print("CSV shape:", len(rows), "x", len(rows[0]), "| header:", rows[0][:3])
        # 첫 열이 뉴런 ID(문자열)라고 가정하고 숫자 부분만 추출
        body = rows[1:]
        adj = np.array([[float(x) for x in r[1:]] for r in body], dtype=np.float32)
        assert adj.shape[0] == adj.shape[1], f"정사각 행렬 아님: {adj.shape}"
        np.savez_compressed(npz_path, adj=adj)
    n = adj.shape[0]
    src, tgt = np.nonzero(adj)
    w = np.asarray(adj[src, tgt], dtype=np.float32)
    return {
        "n": n,
        "edge_index": torch.tensor(np.stack([src, tgt]), dtype=torch.int64),
        "syn_count": torch.tensor(w, dtype=torch.float32),
        "signs": None,  # [미확인] 유충 E/I 부호 별도 파일 여부는 나중에 확인
    }


def load_flyvis_builtin() -> dict:
    """flyvis 패키지 내장 커넥톤 JSON (45,669뉴런 visual system)."""
    import flyvis
    from flyvis.connectome import ConnectomeFromAvgFilters  # 설치 후 경로 확정 예정
    # 실제 로딩 코드는 flyvis 설치 완료 후 API 확인해서 채움
    raise NotImplementedError("flyvis 설치 완료 후 구현 — install_notes.md 참조")


if __name__ == "__main__":
    c = load_larval()
    e = c["edge_index"]
    n = c["n"]
    dens = e.shape[1] / (n * (n - 1))
    print(f"larval: n={n:,} edges={e.shape[1]:,} conn_density={dens*100:.3f}% syn_sum={c['syn_count'].sum():,.0f}")
