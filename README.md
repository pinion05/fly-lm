# fly-lm: Can a Fly Brain Learn Language?

**Full adult fly CNS connectome as a fixed recurrent architecture for char-level language modeling — trained from scratch.**

> 한 마디로: 초파리 수컷 전체 중추신경계(MaleCNS v1.0)의 실제 배선을
> 고정된 순환신경망 구조로 쓰고, TinyStories로 다음 글자 예측만 학습시켰다.
> 실제 배선은 언어 통계를 배우고, 같은 조건의 무작위 배선은 배우지 못한다.

## 핵심 결과 (동일 배치 40개 공정평가, seed 123)

| 배선 | eval loss (nats/char) | PPL | unigram(3.080) 대비 |
|---|---|---|---|
| **실제 파리 배선 (MaleCNS)** | **2.629** | 13.9 | -0.45 |
| 차수보존 재배선 (Maslov–Sneppen) | 2.952 | 19.1 | -0.13 |
| 완전 무작위 배선 | 3.112 | 22.4 | +0.03 (unigram에 갇힘) |

- 파리 배선은 학습 파라미터 160만 개(셀타입 쌍별 시냅스 세기 α + bias)로
  unigram을 확실히 깨고 문맥 학습에 진입
- 무작위 배선은 8,000스텝 내내 unigram 수준에 정체 — **학습 자체가 성립 안 함**
- 차수보존 null(허브 구조 100% 유지, 연결 패턴만 교차)도 실배선에 못 미침
  → **학습 가능성의 대부분은 허브가 아니라 세포타입 간 연결 패턴에서 나옴**

## 실험 설정

- **커넥톤**: MaleCNS v1.0 (brain + nerve cord, 수컷 성충 전체 CNS)
  - 그래프: 뉴런 211,577 / 연결 10,653,945 (weight≥3 컷) / 시냅스 합 1.25억
  - E/I 부호: 뉴런별 신경전달물질 예측(GABA→억제) 실측 매핑
  - 셀타입 11,752종 → 관측 타입쌍 α ~152만 파라미터 (flyvis weight-sharing의 확장)
- **과제**: TinyStories 60M chars, char-level next-token (vocab 107)
- **학습**: Adam lr 1e-3, BPTT truncation (warm 32 + train 32), 8,000 steps, 단일 GPU (임대)
- **3-way 대조**: 실배선 vs Maslov–Sneppen 차수보존 재배선 vs 독립 셔플 무작위
  (엣지 수·세기 분포·E/I 비율·파라미터 수·학습 설정 전부 동일)

## 스테이지 히스토리

| Stage | 내용 | 결과 |
|---|---|---|
| 0 | 유충 2,952뉴런 스모크 (CPU) | 파이프라인 검증, unigram 승 |
| 0' | flyvis visual 45,669뉴런 | loss 3.10 — unigram 근처, 시각회로는 언어에 부적합 확인 |
| 1 | flyvis 45,669 + 무작위 대조 | 무작위 2.88 < 파리 3.09 — 시각 커넥톤에선 역전 (관찰) |
| 2 | **MaleCNS 전체 + 무작위 대조** | **파리 2.629 ≫ 무작위 3.112** |
| 2.5 | **Maslov–Sneppen 차수보존 대조** | 파리 2.629 > 차수보존 2.952 — 허브만으로 설명 불가 |

## 저장소 구성

```
connectome_data.py   유충 커넥톤 로더 (Winding 2023 S1)
flyvis_loader.py     flyvis 내장 visual 커넥톤 로더
prep_malecns.py      MaleCNS feather → 학습용 npz 전처리
celltype_model.py    셀타입 쌍별 α 모델 (flyvis 방식 확장)
smoke_larval.py      Stage 0 스모크
train_flyvis.py      Stage 0' 학습 (truncated BPTT)
smoke_flyvis.py      Stage 1 모델/학습
vast_stage2.sh       Stage 2 전체 파이프라인 (Vast.ai)
maslov_rewire.py     Maslov–Sneppen 차수보존 재배선 + 검증
dashboard.py         실시간 학습 대시보드 (테일넷 전용)
stage{1,2}_results.json / maslov_results.json  최종 수치
stage{2}_train.log / train_maslov.log          전체 학습 로그
stage3_design.py     (예정) 파리뇌 계층 래핑 설계
```

## 재현

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy
# 커넥톤: MaleCNS v1.0 공개 덤프 (CC BY 4.0)
#   https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
#   body-annotations, body-neurotransmitters 동일 경로
# 데이터: https://huggingface.co/datasets/roneneldan/TinyStories
python prep_malecns.py   # → malecns_graph.npz
python maslov_rewire.py  # → malecns_maslov.npz (차수보존 대조)
# 학습: vast_stage2.sh 내 train_stage2.py 참조 (B2/TBPTT32/8000 steps)
```

하드웨어: RTX 5070 Ti / 4080 SUPER 임대로 충분 (16GB VRAM, ~9.3GB 사용).
전체 프로젝트 GPU 비용: **$3 미만** (3-way 대조 포함).

## 학계 위치 (요약)

- "생물 배선 > 무작위" 자체는 정설 — Suárez 2021 (Nat. Mach. Intell., 인간 macro 커넥톰),
  Shiu 2024 (Nature, 시뮬레이션), Drosophila Connectome Reservoir 2025 (Biomimetics, 같은 MaleCNS+ESN)
- 본 저장소의 신규 기여: (1) 전체 CNS 규모의 gradient(BPTT) 학습 — 최대 선행 45.7k 뉴런,
  (2) 언어 과제로의 확장, (3) 무작위 대조의 실패가 unigram 정체라는 정량 서술,
  (4) 차수보존 null로 허브 가설 기각
- 상세 문헌 조사: `fly-lm-literature-position-report.md` (1차 소스 검증 포함)

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
- 학습 파라미터화(타입쌍 α)가 무작위 대조에서 갖는 역할의 분리 분석 미완

## 로드맵

- [ ] 시드 3개 반복 (±$15)
- [ ] spectral radius·활동 분산 분석
- [ ] 50k 스텝 수렴 실험 (loss ~2.0 목표)
- [ ] 파리뇌 계층 래핑 (Stage 3)
- [ ] arXiv 프리핀트 → ICLR 2027 워크숍 (Re-Align 계열)
