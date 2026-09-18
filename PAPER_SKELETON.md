# 골격: 생물 배선의 학습 가능성 — 전체 초파리 커넥톰을 아키텍처로 쓴 언어 모델

> 상태: 1차 골격 (2026-09-18). 모든 수치는 실측 확정. [추론] 태그는 해석이며 본문에서 명시됨.
> 목표 게재: ICLR 2027 워크숍 (Re-Align 계열, 2027-02-01 마감 예상) — 5페이지 논아카이벌.

---

## 제목 후보

1. **Wiring Weight, Not Wiring Pattern: A Complete Insect Connectome as a Fixed Recurrent Architecture for Language Modeling**
2. The Fly Brain Reads TinyStories: Structural Learnability of a Complete Connectome
3. What a Whole-Brain Connectome Buys You: Weight Allocation, Not Topology, Drives Learnability

- 후보 1 추천: MWS 실험 후 핵심 발견이 제목에 직결. "Wiring Weight, Not Wiring Pattern"이 한 줄 요약.

## 초록 (확정 수치 반영판)

- 문장 1 (갭): 전체 동물 커넥톰을 gradient 학습 아키텍처로 쓴 적은 없다.
- 문장 2 (방법): MaleCNS v1.0 (211,577 뉴런, threshold 필터 후 10.65M 엣지)을 고정 sparse recurrent 구조로 동결, cell-type-pair 시냅스 스케일 α 1.6M + 임베딩/readout만 학습. TinyStories char-level next-token prediction.
- 문장 3 (3-way): 실배선 2.225±0.118 / 차수보존 Maslov 재배선 2.550±0.047 / 무작위 3.112 (unigram 3.080으로 붕괴). 격차는 시드 분포 완전분리 + Welch p=0.028.
- 문장 4 (핵심 반전): strength 보존 재배선(위상 파괴, 노드별 시냅스 무게 보존)은 실배선과 통계적 동급(2.145±0.050, Welch t=-1.09)으로 회복 → 학습 가능성은 배선 패턴이 아니라 **허브 위계에 대한 가중치 분배**에서 나온다.
- 문장 5 (부 발견): stateful BPTT에서 실배선만 h 궤적 분포에 조건화된 shortcut으로 붕괴(state-level exposure bias) — 차수보존 null는 멀쩡.
- 문장 6 (효율): 총 노출 토큰은 데이터셋의 0.3%, TinyStories 원논문 총학습량 대비 ~0.001%.
- 문장 7 (릴리즈): 코드·배선 파일·스텝별 로그 전부 공개.

---

## 1. Introduction (0.75p)

- **P1 갭**: 커넥톰 시대 — C. elegans(302), larval fly(→ 성체 MaleCNS 전체), flyvis(Nature 2024)는 시각 과제. 전체 CNS를 gradient로 프리트레인한 사례 없음.
- **P2 질문**: "생물 배선은 학습 가능한 아키텍처인가?" — 무엇이 학습 가능성을 결정하나? 후보 3개: ①위상(routing) ②차수 분포(허브) ③가중치-허브 상관(무게 분배).
- **P3 기여 4개**:
  1. 최초의 전체 커넥톰 스크래치 프리트레인 (211k 뉴런, 언어 과제)
  2. 3-way + strength 보존 4-way 대조 설계로 "학습 가능성의 원천"을 분리
  3. 발견: 가중치 분배가 원천, 위상은 아님 (MWS 실험)
  4. 부 발견: state-level exposure bias — 좋은 배선일수록 상태 의존 shortcut에 취약

## 2. Related Work (0.5p)

- **커넥톰→계산**: Shiu 전뇌 LIF (시냅스 수 초기화), flyvis 45k 시각, Suárez 2021 (위상 주장의 선례 — 우리는 이 논지를 정정하는 쪽).
- **생물 영감 구조**: reservoir computing / ESN (고정 무작위 배선), Liquid machines. 우리와의 차이: 배선이 생물 실측 + 학습 파라미터가 시냅스 쌍 단위.
- **n-gram 사다리**: char-LM 문헌상 TinyStories 공표 baseline 부재 → 우리 실측(unigram 3.081 / bigram 2.297 / trigram 1.683)이 참조 표.
- **shortcut learning**: Geirhos 2020, exposure bias (Bengio 2015 scheduled sampling) — 토큰 수준→상태 수준 확장.

## 3. Methods (1.0p)

- **3.1 커넥톰 전처리**: MaleCNS v1.0 feather (1,051,241,946 bytes) → 시냅스 수 ≥3 컷 → 10,653,945 엣지 / 211,577 뉴런. E-I 부호 (+21.16M/−4.87M 시냅스). cell type 11,752종 → 쌍(pair) 1,519,846.
- **3.2 모델**: leaky-RNN, h←0.5h+0.5tanh(Wh+b), clamp ±10. 학습: α(cell-type-pair 스케일, clamp [0,3]) + type_bias + sensory 임베딩(107→256) + motor readout(512→107). sensory 주입 256 / motor 판독 512 슬롯은 고정 시드 무작위 샘플. 총 학습 파라미터 ~8M (그래프 자체 10.65M 엣지는 동결).
- **3.3 대조 설계 (4-way)**:
  - FLY: 실배선
  - MASLOV: Maslov–Sneppen 차수보존 재배선 — 위상 파괴 + 가중치-허브 상관 파괴
  - MWS: strength 보존 재배선 (신규) — 같은 (syn,sgn) 클래스 내 스왑 → 노드별 in-degree·in-strength 정확 보존, 위상만 파괴 (상호성 9.54%→0.28%)
  - RANDOM: 완전 무작위 재배선
- **3.4 학습 프로토콜**: TBPTT 32 + warm 32 (h=0), B=2, Adam 1e-3, clip 1.0, 20k 스텝, 시드 3회 (0/1/2). 공정평가: 항상 eval seed 123, 40배치.
- **3.5 그래프 분석**: 스펙트럼(eigs), modularity, BFS 거리, 상호성, 허브 제거 — 전부 실측.

## 4. Results (1.5p)

- **4.1 3-way 메인** (Table 1):
  - FLY 2.225±0.118 > MASLOV 2.550±0.047 > RANDOM 3.112 ≈ unigram 3.080
  - 시드 완전분리 (FLY 최악 2.332 < MASLOV 최선 2.497), Welch p=0.028
  - 8k→20k (2.5× 학습)에도 격차 유지 → 초기 수렴 속도 아니라 지속 구조 효과
  - 학습 곡선 그림 (Fig 1)
- **4.2 가중치 분배 vs 위상 분해** (핵심 신규, Table 2):
  - MWS 2.145±0.050 — FLY와 통계적 동급 (t=-1.09), MASLOV와 압도적 차이 (t=-10.29)
  - 대응 구조 실측: strength 상위 1% 독점 FLY 20.35% = MWS 보존, MASLOV 12.88% 파괴 / λ₁ FLY 5976, MWS 3518, MASLOV 1999 / 상호성 9.54% vs 0.28% vs 0.14%
  - 해석: Maslov null은 두 요인을 혼재 — 차수보존만으로는 "허브 가설 기각"이라 부를 수 없고, "가중치-허브 상관 파괴"가 진짜 타격. 리뷰어용 방법론 포인트.
- **4.3 성능 위치** (n-gram 사다리):
  - FLY 2.225 ≈ bigram 2.297 하회~동급, trigram 1.683에는 못 미침
  - bigram 카운트는 3,000만 자, 우리 노출 ~300만 자 → 10분의 1 데이터로 bigram 도달 [샘플 효율]
  - TBPTT-32(≈5-6단어) 지평과 성능 차수의 정합 → 상한은 데이터가 아니라 수용창 [추론]
- **4.4 stateful BPTT의 함정** (부 발견):
  - 경계정렬+stateful: FLY 학습 loss 2.27 (개선) 그러나 h=0 재시작 시 교사강제 CE 5.7~6.6 (unigram 2배 나쁨), 'e' 82~97% 붕괴
  - MASLOV는 h=0 멀쩡 (2.599) — 비대칭
  - 반증: 워밍업 100→5,000글자로도 미회복 → "숨은 기억" 아니라 h 분포 조건화 shortcut
  - 명명: state-level exposure bias (토큰 수준 기존 개념의 상태 수준 인스턴스)
  - S5: 경계정렬 단독은 무차이 (FLY 2.327) — stateful이 붕괴의 필요조건
- **4.5 샘플 품질 타임랩스** (Fig 2): 8k→20k 진화, 관사·구두점 리듬 형성

## 5. Discussion (0.75p)

- **주 해석**: 학습 가능성의 원천 = 가중치-허브 상관. 허브 위계(중앙값 27, 최대 9,193)에 시냅스 무게가 앉히는 방식이 다중 시정수 reservoir를 만든다 [해석].
- **커넥톰이 안 주는 것**: 기능적 시냅스 효능 — 우리 α는 cell-type 쌍 단위 공유라 개별 시냅스 해상도 아님. 생물 학습 규칙과 무관.
- **한계 명시** (오버클레임 방어선):
  1. "파리 뇌가 언어를 이해한다" 아님 — 커넥톰 위상의 시퀀스 학습 이점을 언어 과제로 측정한 것 (Suárez 2021과 동일 선)
  2. 절대 성능은 bigram 수준 — 아키텍처 수용창(TBPTT-32) 제약
  3. 시드 분산 주장(FLY σ가 2.5배)은 n=3으로 방어 불가 (p=0.137) — 강등 서술
  4. 학습량 0.3%/0.001% — 데이터 기아 상태의 결과임을 명시
  5. 단일 과제(char-LM) — 다른 도메인 일반성 미검증
- **뉴로AI 함의**: 커넥톰을 아키텍처 사전(prior)으로 쓰는 실용 지침 — 전체를 다 쓸 필요 없이 가중치 분배 통계만 보존하면 될 수도.

## 6. Conclusion (0.25p)

- 한 문장 요약: 전체 커넥톰은 학습 가능하며, 그 이점은 배선 패턴이 아니라 허브 위계의 가중치 분배에서 나온다.
- Future: word-level 제한 어휘 코퍼스, TBPTT 확장(수용창 검증), 시드 n≥10, 다중 과제.

## 부록 / 릴리즈

- 코드: github.com/pinion05/fly-lm (전 스테이지 커밋 히스토리)
- 배선 파일: malecns_graph.npz / maslov / mws (생성 코드 포함)
- 스텝별 로그·결과 JSON 전부
- 재현 비용: 런당 ~$0.4 (Vast.ai RTX 5070 Ti)

---

## 표/그림 목록

- **Table 1**: 4-way × 3시드 eval (FLY/MWS/MASLOV/RANDOM + unigram) — 본 논문의 중심 표
- **Table 2**: 구조 지표 대응 (strength 독점 / λ₁ / 상호성 / 모듈러리티 / 깊이)
- **Table 3**: n-gram 사다리 (unigram/bigram/trigram vs FLY)
- **Fig 1**: 학습 곡선 (기존 training_curves.png 갱신 — MWS 곡선 추가)
- **Fig 2**: 샘플 타임랩스 (8k→20k)
- **Fig 3** (선택): MWS 개념도 — 같은 무게, 다른 배선

## 집필 분업

- Hermes: 영어 초안 전 섹션 → 니 검토
- 박명철: 서사 방향 승인, 이해 안 되는 문장 질문 (검증 역할), acknowledgement AI 기재 여부 결정
- 순서: Methods(사실 기술이라 빠름) → Results → Intro → Discussion → 초록 마지막 다듬기
