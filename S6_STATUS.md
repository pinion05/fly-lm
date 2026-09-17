# Stage 6 시드 반복 — 실행 상태 (2026-09-17 19:55 KST, s6-watch 크론용)

## 완료·회수·destroy 됨 (SSH 재시도 ❌ — 인스턴스 없음)
- FLY_s1 (51284230): ALL-DONE, s6=2.2437 → s6_FLY_seed1_results.json + train_s6_FLY_s1.log 회수 완료, destroy 완료
- MASLOV_s1 (51284469): ALL-DONE, s6=2.4967 → s6_MASLOV_seed1_results.json + train_s6_MASLOV_s1.log 회수 완료, destroy 완료
- FLY_s2 (51284607): ALL-DONE, s6=2.0989 → s6_FLY_seed2_results.json + train_s6_FLY_s2.log 회수 완료, destroy 완료

## 진행 중 (유일 잔여)
- MASLOV_s2 (51285180, ssh9.vast.ai:15180): 19:52 기준 step 12600/20000, ~0.87s/step → 완료 예상 ~21:40 KST
- 로그 /root/work/train_s6_MASLOV_s2.log, 결과 /root/work/s6_MASLOV_seed2_results.json

## 다음 크론 실행 절차 (MASLOV_s2만 처리)
1. ssh -o ControlMaster=no -o ControlPath=none -i ~/.ssh/id_vastai -p 15180 root@ssh9.vast.ai 로 train_s6_MASLOV_s2.log ALL-DONE 확인
2. 진행 중이면 step/loss 한 줄 보고 후 종료
3. ALL-DONE이면: 결과 JSON + 로그 scp 회수 → 통계 → destroy instance 51285180 -y → 최종 커밋+push → s6-watch 크론 remove

## 확정 통계 (seed0은 기존 기록값)
- FLY 3시드: 2.332 / 2.2437 / 2.0989 → mean 2.2249, std 0.1177
- MASLOV 2시드(부분): 2.569 / 2.4967 → mean 2.5328, std 0.0512 (s2 대기)
