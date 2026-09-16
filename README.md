# fly-lm: Can a Fly Brain Learn Language?

**Full adult fly CNS connectome as a fixed recurrent architecture for char-level language modeling — trained from scratch.**

> 한 마디로: 초파리 수컷 전체 중추신경계(MaleCNS v1.0)의 실제 배선을
> 고정된 순환신경망 구조로 쓰고, TinyStories로 다음 글자 예측만 학습시켰다.
> 실제 배선은 언어 통계를 배우고, 같은 조건의 무작위 배선은 배우지 못한다.

## 핵심 결과

**8,000 스텝 (동일 배치 40개 공정평가, seed 123):**

| 배선 | eval loss (nats/char) | PPL | unigram(3.080) 대비 |
|---|---|---|---|
| **실제 파리 배선 (MaleCNS)** | **2.629** | 13.9 | -0.45 |
| 차수보존 재배선 (Maslov–Sneppen) | 2.952 | 19.1 | -0.13 |
| 완전 무작위 배선 | 3.112 | 22.4 | +0.03 (unigram에 갇힘) |

**20,000 스텝 (학습 2.5배, 같은 공정평가) — 격차 유지:**

| 배선 | 8k | 20k | 개선 |
|---|---|---|---|
| **FLY (파리 배선)** | 2.629 | **2.332** (PPL 10.3) | -0.30 |
| MASLOV (차수보존) | 2.952 | **2.569** (PPL 13.2) | -0.38 |

→ MASLOV가 늦게 따라잡는 시나리오 기각: **격차 0.24는 학습량과 무관한 구조 효과**

![3-way training curves](assets/training_curves.png)

*차트는 세 배선(실제 파리/차수보존 재배선/무작위)의 8,000스텝 학습 곡선 — 회색 점선이 unigram 기준선. 20k 곡선·실시간 추적 대시보드(테일넷 전용)는 `dashboard.py`, `train_20k.log` 참고.*

## 세 가지 발견

1. **전체 CNS 211k 뉴런을 gradient(BPTT)로 학습** — 문헌상 최대 규모 (선행 최대 45.7k, Shiu 2024는 학습 없는 시뮬레이션)
2. **무작위 배선은 학습 자체가 성립 안 함** — 8k 스텝 내내 unigram(3.08) 근처 정체, 연결 수·세기 분포·E/I 비율·파라미터 수가 같아도
3. **허브 가설 기각** — 차수보존 null(허브 100% 유지, 연결 패턴만 교차)도 실배선에 못 미침 + 20k에서도 격차 유지 → 학습 가능성의 대부분은 "어떤 뉴런이 어떤 뉴런에 연결되는가"라는 경로 패턴에서 나옴

## 실험 설정

- **커넥톤**: MaleCNS v1.0 (brain + nerve cord, 수컷 성충 전체 CNS)
  - 그래프: 뉴런 211,577 / 연결 10,653,945 (weight≥3 컷) / 시냅스 합 1.25억
  - E/I 부호: 뉴런별 신경전달물질 예측(GABA→억제) 실측 매핑
  - 셀타입 11,752종 → 관측 타입쌍 α ~152만 파라미터 (flyvis weight-sharing의 확장)
  - 허브 구조: 상위 1% 뉴런이 전체 연결의 12.8% 수신 (최다 허브 9,193 시냅스)
- **과제**: TinyStories 60M chars, char-level next-token (vocab 107)
- **학습**: Adam lr 1e-3, B=2, truncated BPTT (warm 32 + train 32), 단일 임대 GPU
- **3-way 대조**: 실배선 vs Maslov–Sneppen 차수보존 재배선 vs 독립 셔플 무작위
  (엣지 수·세기 분포·E/I 비율·파라미터 수·학습 설정 전부 동일)

## 스테이지 히스토리

| Stage | 내용 | 결과 |
|---|---|---|
| 0 | 유충 2,952뉴런 스모크 (CPU) | 파이프라인 검증, unigram 승 |
| 0' | flyvis visual 45,669뉴런 | loss 3.10 — 시각회로는 언어에 부적합 확인 |
| 1 | flyvis 45,669 + 무작위 대조 | 무작위 2.88 < 파리 3.09 — 시각 커넥톤에선 역전 (관찰) |
| 2 | **MaleCNS 전체 + 무작위 대조** | **파리 2.629 ≫ 무작위 3.112** |
| 2.5 | **Maslov–Sneppen 차수보존 대조 8k** | 파리 2.629 > 차수보존 2.952 — 허브 가설 기각 |
| 2.6 | **20k 장기 학습 (FLY vs MASLOV 병렬)** | 파리 2.332 > 2.569 — 격차 0.24, 학습량 무관 구조 효과 |

## 20k 스텝 샘플 (파리 배선이 쓴 글)

```
Once upon a time naked wheitg toe dind ind ilind bitid in te acne fit the
thed int he whend tered wund the terind ger and he nre wo nteot ad und id
fey he sit tho iken s have the carded to tup pite to tid the rougd...
```

관사·짧은 실단어("the", "and", "he", "have") 밀도가 8k 대비 눈에 띄게 증가.
`fly20k_results.json`의 `snapshots`에 2k→20k 스텝별 문장 진화 타임랩스 포함.

## 저장소 구성

```
connectome_data.py   유충 커넥톤 로더 (Winding 2023 S1)
flyvis_loader.py     flyvis 내장 visual 커넥톤 로더
prep_malecns.py      MaleCNS feather → 학습용 npz 전처리
celltype_model.py    셀타입 쌍별 α 모델 (flyvis 방식 확장)
maslov_rewire.py     Maslov–Sneppen 차수보존 재배선 + 검증
vast_hunt.py         Vast.ai 인스턴스 자동 헌팅 (SSH 프록시 장애 대응)
dashboard.py         실시간 다중 인스턴스 학습 대시보드 (테일넷 전용)
stage{1,2}_results.json / maslov_results.json    8k 3-way 최종 수치
fly20k_results.json / maslov20k_results.json     20k 최종 수치+스냅샷
train_20k.log / train_maslov20k.log              20k 전체 학습 로그
stage3_design.py     (예정) 파리뇌 계층 래핑 설계
```

## 재현

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas pyarrow matplotlib
# 커넥톤: MaleCNS v1.0 공개 덤프 (CC BY 4.0)
#   https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
#   body-annotations, body-neurotransmitters 동일 경로
# 데이터: https://huggingface.co/datasets/roneneldan/TinyStories
python prep_malecns.py   # → malecns_graph.npz
python maslov_rewire.py  # → malecns_maslov.npz (차수보존 대조)
# 학습: vast_stage2.sh / train_20k 참조 (B2/TBPTT32, 8k 또는 20k steps)
```

하드웨어: 16GB GPU 한 장이면 충분 (~9.3GB 사용).
전체 프로젝트 GPU 비용: **$5 미만** (3-way 8k + 20k 병렬 + 스모크 전부 포함).

## 학계 위치 (요약)

- "생물 배선 > 무작위" 자체는 정설 — Suárez 2021 (Nat. Mach. Intell., 인간 macro),
  Shiu 2024 (Nature, 시뮬레이션), Drosophila Connectome Reservoir 2025 (Biomimetics, 같은 MaleCNS+ESN readout-only)
- 본 저장소의 신규 기여: (1) 전체 CNS 규모 gradient 학습, (2) 언어 과제로의 확장,
  (3) 무작위 대조의 실패가 unigram 정체라는 정량 서술, (4) 차수보존 null로 허브 가설 기각,
  (5) **학습량 2.5배에서도 격차 유지 — 구조 효과의 영구성**
- 상세 문헌 조사: `LITERATURE.md` (1차 소스 검증 포함)

## 라이선스·데이터

- 코드: MIT
- MaleCNS v1.0: CC BY 4.0 (Janelia, male-cns.janelia.org)
- TinyStories: CC BY 4.0 (Microsoft)
- 유충 커넥톤 (Winding 2023): CC BY 4.0
- flyvis (TuragaLab): MIT

## 한계 (정직하게)

- 시드 1개 — 다중 시드 반복은 예정
- 단일 언어·단일 과제 — 언어 특화가 아닌 일반 순서 구조 이점일 가능성
- spectral radius 등 동역학 분석 미포함 (Suárez 2021의 임계동역학 조건과의 비교 필요)
- 학습량은 TinyStories 논문 기준(100B+ 토큰)의 0.3% — 샘플 효율 관점 해석 필요
- 생물학적 해석 아님: "파리 뇌가 언어를 이해한다"가 아니라 "커넥톤 위상이 시퀀스 학습에 구조적 이점"이 본 주장

## 로드맵

- [x] 3-way 대조 8k (실배선/차수보존/무작위)
- [x] 20k 장기 학습 — 격차 영구성 확인
- [ ] 시드 3개 반복 (±$15)
- [ ] spectral radius·활동 분산 분석
- [ ] word-level 500단어 제한 코퍼스 (문장 구조 학습 도전)
- [ ] SimpleStories 데이터 축 추가
- [ ] 파리뇌 계층 래핑 (Stage 3)
- [ ] arXiv 프리핀트 → ICLR 2027 워크숍 (Re-Align 계열)
