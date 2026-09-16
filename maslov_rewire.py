"""Maslov–Sneppen 차수보존 재배선 (degree-preserving rewiring).

목적: Stage 2의 "파리 배선 > 무작위 배선" 결과에 대한 반론 차단.
반론: "무작위 배선이 못 한 건 허브(차수 분포)가 파괴됐기 때문일 수 있다"
대응: in/out 차수 시퀀스를 정확히 보존한 채 위상만 뒤섞은 null을 만들어 비교.
  → 파리가 이 배선도 이기면 "허브 구조만이 아니라 연결 패턴 자체가 정보"라는 주장 성립.

알고리즘 (Maslov & Sneppen 2002):
  반복: 무작위 엣지 2개 (a→b, c→d) 선택 → (a→d, c→b)로 교환
  조건: a≠c, 새 엣지들이 자기루프/중복 아니어야 교체 채택
  엣지당 평균 ~10회 시도면 위상은 충분히 섞이고 차수는 100% 보존.

세기(syn)/부호(sgn)는 엣지에 붙은 채로 이동 (각 연결의 물성 유지).
셀타입(type_of_node)은 노드에 붙어있으므로 자동으로 재조합됨 —
즉 "타입쌍 alpha"의 자유도도 실측 타입쌍 구조를 따라감.
"""
import numpy as np
import time

rng = np.random.default_rng(2026)

d = np.load("/root/work/malecns_graph.npz")
src, tgt = d["src"], d["tgt"]
syn, sgn = d["syn"], d["sgn"]
E = len(src)
print(f"E={E:,}")

# 중복 체크용 set (python int pair) — 10.6M이면 ~3GB RAM, 서버 62GB라 OK
edge_set = set(zip(src.tolist(), tgt.tolist()))
print("edge set built", flush=True)

t0 = time.time()
attempts = E * 10
swaps = 0
for i in range(attempts):
    j1 = rng.integers(0, E)
    j2 = rng.integers(0, E)
    a, b = src[j1], tgt[j1]
    c, dd = src[j2], tgt[j2]
    if a == c:
        continue
    # 새 엣지: (a, dd), (c, b)
    if a == dd or c == b:
        continue
    e1, e2 = (a, dd), (c, b)
    if e1 in edge_set or e2 in edge_set:
        continue
    # 교체
    edge_set.discard((a, b))
    edge_set.discard((c, dd))
    edge_set.add(e1)
    edge_set.add(e2)
    src[j1], tgt[j1] = a, dd
    src[j2], tgt[j2] = c, b
    swaps += 1
    if i % (E) == 0 and i > 0:
        print(f"  {i:,}/{attempts:,} 시도, {swaps:,} 교환, {time.time()-t0:.0f}s", flush=True)

print(f"done: {swaps:,} swaps in {time.time()-t0:.0f}s (swaps/edge = {swaps/E:.2f})", flush=True)

# 검증: 차수 보존 확인
in_orig = np.bincount(d["tgt"], minlength=int(d["n"]))
in_new = np.bincount(tgt, minlength=int(d["n"]))
out_orig = np.bincount(d["src"], minlength=int(d["n"]))
out_new = np.bincount(src, minlength=int(d["n"]))
assert np.array_equal(in_orig, in_new), "in-degree 변경됨!"
assert np.array_equal(out_orig, out_new), "out-degree 변경됨!"
print("degree 보존 검증: PASS (in/out 모두 동일)", flush=True)

np.savez_compressed("/root/work/malecns_maslov.npz",
                    n=d["n"], src=src, tgt=tgt, syn=syn, sgn=sgn,
                    type_of_node=d["type_of_node"], n_types=d["n_types"],
                    input_idx=d["input_idx"], output_idx=d["output_idx"])
print("saved malecns_maslov.npz", flush=True)
print("MASLOV-DONE", flush=True)
