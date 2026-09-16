"""MaleCNS v1.0 전처리: feather → 학습용 npz (Vast.ai 인스턴스에서 실행 가정).

입력 (male-cns.janelia.org GCS 공개):
  connectome-weights-male-cns-v1.0-minconf-0.5.feather  (전체 연결 그래프, 1.05GB)
  body-annotations-male-cns-v1.0-minconf-0.5.feather    (뉴런 주석: 타입/층/NT 등)
  body-neurotransmitters-male-cns-v1.0.feather          (신경전달물질 예측)

출력 malecns.npz:
  n, src, tgt, syn, sgn(E/I 부호: NT 기반), type_of_node(cellType 코드),
  n_types, input_idx(감각), output_idx(운동/하강)
"""
import numpy as np
import pandas as pd

D = "/root/work/"

cw = pd.read_feather(D + "connectome-weights.feather")
print("connectome:", cw.shape, cw.columns.tolist())
print(cw.head(3).to_string())

ann = pd.read_feather(D + "body-annotations.feather")
print("\nannotations:", ann.shape)
print(ann.columns.tolist())

nt = pd.read_feather(D + "body-neurotransmitters.feather")
print("\nnt:", nt.shape, nt.columns.tolist())
