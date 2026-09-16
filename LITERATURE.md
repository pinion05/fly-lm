# 커넥톤 구조 → 학습 가능성(learnability): 학계적 위치 조사 보고서

조사일: 2026-09-15 (KST). 모든 핵심 주장은 1차 소스(논문 전문 grep 또는 API 메타데이터)로 검증. 미검증 항목은 [미확인] 표기.

## 한 줄 결론

**판정: (ii) 부분적으로 새로움.** "생물 배선이 무작위 배선보다 계산에 유리하다"는 주장 자체는 **이미 정설** (Suárez 2021 Nat. Mach. Intell.이 정본). 새로운 것은 조합이다: **전체 CNS 211k 뉴런 + BPTT gradient 학습 + 언어 과제 + 매칭된 무작위 대조군의 정량적 실패(unigram 정체)**. 단, 우리 대조군이 차수 분포(degree sequence)를 보존하지 않았다면 약점이 있음 — §5 참고.

---

## 1. 직접 선행연구: 커넥톰을 ANN으로 쓰는 연구

| 연구 | 생물 vs 무작위 비교? | 결과 |
|---|---|---|
| **flyvis** — Lappalainen et al. 2024, *Nature* 634:1132–1140, [DOI 10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3) (전문 PMC11525180 grep: `shuffl`/`randomized`/`random connectome`/`control model` = **0 hits**) | **없음**. 비교 대상은 unconstrained CNN(하한 추정)과 MNIST sparsity probe뿐 | 시각 과제(optic flow)에서 커넥톤 제약 모델 ≈ unconstrained CNN. 언어 아님 |
| **Shiu et al. 2024**, *Nature*, [DOI 10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9) (PMC11446845 전문 확인) | **있음 — weight shuffle 대조**. "connectivity weights were shuffled randomly (while maintaining the global connectivity weight distribution)" | 정확한 connectome: 당 설탕→MN9 운동뉴런 활성화 **100/100 시뮬레이션 성공**, shuffle 대조는 **1/100만 성공**. 단, 학습 아닌 고정가중치 LIF 시뮬레이션의 기능 발현 |
| **Kotar & Tuckute 2025**, arXiv [2504.21047](https://arxiv.org/abs/2504.21047) "Model Connectomes" (abs 페이지 실측) | 부분 — 2개의 closely matched control | 단 **"model connectome"은 모델이 생성한 구조, 생물 커넥톰 아님**. 언어(100M 토큰)에서 구조 상속이 저데이터 영역에서 효율적 prior임을 시사. 생물 배선 vs 무작위 비교 아님 |
| Connectome-GPT-Worm, fly-hf, fly-as-a-lm, fly-connectome-lm, nftechie/flm | 일부 readout-only 대조 | 전부 GitHub 전용, 비동행. fly-connectome-lm은 bigram보다 나쁨(리저버 readout만 학습); nftechie/flm은 파리 리저버가 direct-input 대조와 사실상 동점(1.3598 vs 1.3593 nats) — [검증 이력: connectome-lm paper bank, 1차 소스 기반] |
| **Drosophila Connectome Reservoir 2025**, *Biomimetics* 10(5):341, [DOI 10.3390/biomimetics10050341](https://doi.org/10.3390/biomimetics10050341) (PMC12109256 전문 확인) — **가장 가까운 선행연구** | **있음 — 가장 체계적**: random topology+random weights 클래식 대조 + hybrid(connectome 위상+무작위 가중치) + shuffled weights(같은 위상·같은 가중치분포) | MaleCNS 전체(및 부분 그래프) ESN, 시계열 예측. spectral radius 0.99에서 connectome 우위, 주효과는 **과적합 강건성**; 전체 커넥톰 리저버 nRMSE < 2%. 단 **readout만 ridge 회귀 학습, BPTT 아님, 언어 아님**. "shuffling the weights has the same effect as randomizing the topology" — 위상·가중치 상승작용 |

**요점**: flyvis는 무작위 대조가 없고, Shiu는 학습 없는 시뮬레이션이며, 언어 과제 + gradient 학습 + 무작위 대조를 동시에 한 연구는 **없음**.

## 2. 구조 vs 무작위 비교 문헌 (정본)

- **Suárez, Richards, Lajoie, Misic 2021, "Learning function from structure in neuromorphic networks", *Nat. Mach. Intell.* 3:771–786**, [DOI 10.1038/s42256-021-00376-1](https://doi.org/10.1038/s42256-021-00376-1) (피인용 129, bioRxiv v1 전문 실측). **이 주제의 정본.**
  - 인간 dwMRI 커넥톰을 리저버로 구현, memory task 학습.
  - Null 모델: **네트워크 크기·밀도·차수열(degree sequence)·노드별 intrinsic-network 배정을 보존하면서 엣지 쌍을 무작위 재배선**(Maslov–Sneppen 방식).
  - 결과: "connectomes achieve superior performance compared to populations of networks with identical low-level features (density, degree sequence) but randomized high-level topology" — **임계 동역학(critical dynamics) 조건에서**. 배선비용 대비 계산력도 모든 동역학 영역에서 실증 네트워크 우위.
  - 즉 "생물 배선 > (저수준 특성만 맞춘) 무작위 배선"은 **macro-scale에서 이미 확립된 결과**.
- **반대편 증거(주의)**: Xie & Yuille 2019, "Exploring Randomly Wired Neural Networks for Image Recognition", ICCV, [DOI 10.1109/ICCV.2019.00137](https://doi.org/10.1109/ICCV.2019.00137) (피인용 311) — ER/BA/WS 무작위 그래프로 만든 **feed-forward conv** 네트워크가 hand-designed NAS와 경쟁. 해석: 무작위 위상의 열세는 (i) 순환 구조, (ii) 가중치 대부분이 고정·제약될 때, (iii) 시간적 기억 과제에서 두드러지며, 전체 가중치를 학습하는 feed-forward에서는 사라짐. 우리 결과(고정 배선 RNN)는 전자에 해당.
- **Zador 2019, "A critique of pure learning", *Nat. Commun.***, [DOI 10.1038/s41467-019-11786-6](https://doi.org/10.1038/s41467-019-11786-6) (피인용 566) — innate structure가 학습을 대체한다는 프레임 논증. 우리 결과의 해석·서론 인용으로 필수.

## 3. Reservoir computing 문헌: "어떤 위상이 좋은가"

- **Kawai, Park, Asada 2019**, *Neural Networks* 112:15–23, [DOI 10.1016/j.neunet.2019.01.002](https://doi.org/10.1016/j.neunet.2019.01.002) (피인용 164): **small-world 위상이 echo state property와 신호 전파를 향상** — 위상→리저버 품질의 대표 정량 결과.
- **"Reservoir computing quality: connectivity and topology"** 2020, *Natural Computing*, [DOI 10.1007/s11047-020-09823-1](https://doi.org/10.1007/s11047-020-09823-1) (피인용 52).
- **conn2res toolbox** — 2024, *Nat. Commun.*, [DOI 10.1038/s41467-024-44900-4](https://doi.org/10.1038/s41467-024-44900-4) (PMC10803782 전문 확인): 인간 consensus connectome 리저버에서 **임계점(α=1)에서 실증 뇌 네트워크의 memory capacity가 (밀도·차수열 보존) rewired null보다 유의하게 큼** — Suárez 2021 결과의 재현·툴화. "topology of the human brain confers computational function"이라고 명시.
- scale-free ESN 계열 존재 (예: 2006 IJCN "Scale-Free Highly-Clustered ESN", DOI 10.1109/IJCNN.2006.247295) — 정설이라기보다 개별 결과.

**결론**: 리저버 문헌에서 "무작위이기만 한 것보다 구조화된(특히 small-world) 위상이 낫다"는 상당히 확립됨. 단 거의 전부 **readout만 학습**하는 설정.

## 4. 우리 결과의 신규성 평가

| 구성 요소 | 상태 |
|---|---|
| (a) 전체 CNS 규모 (211,577 뉴런) | **새로움**. 학습 포함 최대는 flyvis(~45.7k, 시각계만). Shiu 127k는 학습 없음. Biomimetics 2025가 MaleCNS 전체를 다뤘으나 readout-only |
| (b) 무작위 대조군과의 정량 격차 | **부분적으로 알려짐**. Suárez 2021(인간 macro, memory capacity), Shiu 2024(1/100), Biomimetics 2025(같은 데이터셋, nRMSE) — 단 **gradient 학습(BPTT) 설정에서의 격차는 미보고** |
| (c) 언어 과제 (char-level next-token) | **거의 새로움**. Kotar & Tuckute는 model connectome(비생물). GitHub 시도들은 비동행·부정적. 동행 심사 논문 없음 확인 |
| "무작위 배선이 unigram에 갇힌다"는 실패 모드 서술 | OpenAlex/arXiv 검색에서 **보고 사례 없음** — 언어과제×생물배선 조합 자체가 없었으므로. 이 표현 자체는 새로운 정량 서술로 보임 (단 'unigram 정체'라는 프레이징의 직접 선행연구 부재는 [미확인] — Google Scholar 미실시) |
| 셀타입 쌍별 α(≈152만 파라미터) 학습 방식 | flyvis의 weight-sharing(734 유효파라미터)를 크게 확장한 것 — 스케일·과제 측면에서 새로움 |

## 5. 해석 논쟁: 무작위 배선 실패의 메커니즘

- **허브/차수 분포 파괴**: 문헌에서 논의됨. Suárez 2021의 null은 **차수열을 보존**했기에 그들의 우위는 허브 구조가 아니라 "high-level topology(모듈성 등)"에 기인. 반면 우리 대조군("같은 엣지 수, 같은 세기 분포, 같은 E/I 비율")이 차수열을 보존하지 않았다면, 격차의 상당 부분이 **heavy-tail degree/rich-club 상실**로 설명 가능 — C. elegans rich club: Towlson et al. 2013, *J. Neurosci.* 33:6380, DOI 10.1523/JNEUROSCI.3839-12.2013; connector hub 이론: Bertolero et al. 2018, *Nat. Hum. Behav.* 2:765. **권고: Maslov–Sneppen 차수보존 재배선 대조를 추가하면 Suárez 2021 수준의 대조 강도가 되어 논문 방어가 견고해짐** (방법론 원전: Maslov & Sneppen 2002, *Science* 296:910, DOI 10.1126/science.1065103).
- **위상 자체(모듈성·small-world)**: 확립된 메커니즘 — Kawai 2019(ESP/신호전파), Biomimetics 2025(클러스터링·중복 경로 → 강건성·과적합 저항), Suárez 2021(모듈 구조가 계산적으로 관련 있음, 임계 동역학과의 상호작용).
- **상태 균질화(state homogenization)**: 직접 일치하는 문헌 서술은 찾지 못함 [미확인]. 가장 가까운 것은 임계동역학-echo state 이론(Suárez 2021: 같은 구조도 동역학 영역에 따라 학습능력이 크게 변함; 무작위 균질 그래프는 임계점에서 멀어질 수 있음). **새 해석으로 제안 가능한 부분** — 무작위 대조군의 spectral radius·활성 분산을 측정해 뒷받침하면 기여가 됨.
- **셀타입 쌍 파라미터 무의미화**: 우리 설정 고유의 논점. flyvis는 타입별 α가 타입 공간의 구조(어떤 타입이 어떤 타입에 연결하는지) 위에서만 의미를 가짐 — 무작위 배선에서 이것이 무너진다는 분석은 문헌에 없음 [미확인, 신규 분석 가능].

## 6. 결론 및 게재 처 [추론]

**판정: (ii) 부분적으로 새로움.**
- 핵심 주장("생물 배선 > 무작위")은 정설 — Suárez 2021이 macro-scale에서, Shiu 2024가 뉴런-scale 시뮬레이션에서, Biomimetics 2025가 **같은 MaleCNS 데이터셋**에서 이미 보임. "우리가 처음으로 보였다"고 쓰면 리젝 사유가 됨.
- 새로운 기여: (1) 최대 규모(전체 CNS) gradient 학습, (2) 언어 과제로의 확장, (3) 무작위 대조의 실패가 unigram 수준 정체라는 정량 서술, (4) 셀타입 쌍 파라미터화 학습.
- 필수 보완: (i) 차수보존 재배선 대조(§5), (ii) seed 다중화, (iii) Suárez 2021·Shiu 2024·Biomimetics 2025·Kotar & Tuckute 정면 인용·비교, (iv) 무작위 대조군의 spectral radius 보고(임계동역학 논란 차단).

**게재 처 [추론]**:
1차: arXiv 선공개 후 → **eLife** 또는 **Nature Communications** (Biomimetics 2025·conn2res가 같은 계열에 실렸고, "구조→기능" 서사가 저널 스코프에 부합). 빠른 검증·워크숍 피드백 원하면 **NeurIPS/ICLR NeuroAI 계열 워크숍** (예: NeurIPS "NeuroAI" 워크숍 라인). 순수 ML 강조(언어모델 측면)면 ICLR 메인도 가능하나 대조군 보강 전엔 무리. 저널 순위상 *Biomimetics*(MDPI)는 최후 수단 — 같은 데이터셋 선행연구가 거기 있으므로 인용 없이는 통과 어려움.

---

### 검증 로그
- flyvis PMC11525180 전문: shuffle/randomized/random connectome/control model 0 hits, unconstrained 4 hits (CNN 비교 목적 확인)
- Shiu PMC11446845: shuffle 대조 구절 원문 인용 확인
- Kotar & Tuckute arXiv 2504.21047: abs 페이지에서 제목·저자·초록 실측 ("model connectome" = 생성 구조 확인)
- Suárez 2021: bioRxiv v1 full.txt 617KB 실측 — rewired null 방법론(크기·밀도·차수열·intrinsic-network 보존) 및 결과 문장 원문 확인; OpenAlex 피인용 129
- conn2res PMC10803782: rewired null 결과 문장 확인
- Biomimetics PMC12109256: 대조군 설계·결과 문장 확인
- OpenAlex title-search: small-world/scale-free ESN, connectome reservoir, randomly wired NN(Xie 2019 cit 311) 확인
- Semantic Scholar 미사용(세션 429 이력), Google Scholar 미실시 — "unigram 갇힘" 프레이징 부재는 [미확인] 처리
