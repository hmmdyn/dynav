# 2026-06-12 실험 사이클

**주제**: GPS 노이즈 열화 곡선(실험 1) + mapdrop sweep(실험 2) + map_tokens ablation(실험 3)
**환경**: 5090 머신 (git 클론 아님, 변경사항은 이 커밋에서 동기화)
**결론**: H3(카메라-지도 융합) 입증 · map_dropout_p=0.1 채택 · GAP(1토큰) 유지 · 구조 변경 불필요

---

## Runs

| run id   | 설정                          | 목적                    | ADE↓      | turn↓       | uturn↓    | 체크포인트                    |
| -------- | --------------------------- | --------------------- | --------- | ----------- | --------- | ------------------------ |
| yxmz09i0 | map-only, mapdrop=0.25      | wiizgrmh 재학습 (유실 복구)  | 0.644     | 0.799/0.842 | 0.851     | `outputs/12-17-59/` best |
| qdwvcsgb | full, mapdrop=0.1           | 실험 2 1차 시도            | —         | —           | —         | 없음 (터미널 종료)              |
| u8p6ssmi | full, mapdrop=0.1           | 실험 2 · **채택 기준선**     | **0.610** | 0.781/0.774 | **0.849** | `outputs/13-24-05/` best |
| y1o15cj5 | full, mapdrop=0.1, tokens=9 | 실험 3 (GAP vs 9tokens) | 0.610     | 0.770/0.773 | 0.888     | `outputs/14-58-46/` best |

_obs-only 컨트롤: pkx580d9 (2026-06-11 복구본), val ADE 0.713 (노이즈 불변)_

---

## 실험 1: GPS 노이즈 열화 곡선

- 입력: map 채널에만 기하 노이즈(회전=heading 오차 σ, 평행이동=GPS 오차) 주입
- 평가: val 3,026 전수, 동일 노이즈 시드 (모델 간 공정 비교)
- 결과 파일: `results/2026-06-12/gps_noise_curve_final.{csv,png}`

| val ADE | 0° | 10° | 20° | 30° | 40° | 60° | 80° |
|---------|-----|-----|-----|-----|-----|-----|-----|
| full(u8p6ssmi) | 0.610 | — | 0.614 | — | 0.647 | 0.726 | 0.839 |
| map-only(yxmz09i0) | 0.644 | — | 0.651 | — | 0.705 | 0.806 | 0.926 |
| obs-only(pkx580d9) | 0.713 | — | 0.713 | — | 0.713 | 0.713 | 0.713 |

**해석 B 채택**: full이 전 구간 완만하게 열화 → σ=40°에서 map-only는 obs-only 수렴, full은 0.647로 양쪽 모두보다 낮음 → 카메라가 지도 오염을 보정 = H3(융합) 증거.

**한계**: σ≥60°에서 full도 obs-only 0.713 추월 (blending이지 gating이 아님). 후속 후보: map noise augmentation 확대.

---

## 실험 2: map_dropout_p 0.1 vs 0.25

- u8p6ssmi(0.1) vs full#2(9c2aqmej, 0.25 — 06-11, best ep6)
- **uturn 열세 완전 해소**: 0.849(0.1) vs 0.889(0.25). uturn이 가장 map-의존적 maneuver라 25% 마스킹이 학습을 깎았던 것
- 강건성 손실 없음: Δ(0→40°) 기울기 동일(+0.037 vs +0.038), 절대값은 0.1이 전 레벨 최저
- **결정**: `train_frodobots.yaml` `map_dropout_p: 0.25 → 0.1`
- 결과 파일: `results/2026-06-12/gps_noise_curve_mapdrop01.{csv,png}`

---

## 실험 3: map_tokens 9 vs 1(GAP)

- y1o15cj5(tokens=9) vs u8p6ssmi(GAP=1), 둘 다 mapdrop=0.1·best
- 노이즈 곡선 완전히 겹침 (σ=0/20/40/60/80°: 0.610/0.611/0.646/0.726/0.831 vs 0.610/0.614/0.647/0.726/0.839)
- uturn에서 GAP 우세 (0.849 vs 0.888)
- **결정**: GAP 유지. 근거 교체: "invariance 때문"이 아니라 "다중 토큰이 아무 이득 없음"
- 결과 파일: `results/2026-06-12/gps_noise_curve_tokens9.{csv,png}`

---

## 종합 결론

- **구조 수정 불필요** — H3 입증, uturn 열세 해소, GAP 무해
- 현 기준선: full + EfficientNet GAP + Self-Attn + mapdrop=0.1
- 논문 figure: `gps_noise_curve_final.*` (full=u8p6ssmi · maponly=yxmz09i0 · obsonly=pkx580d9)
- **미착수 후속**: 모델 선택 지표 교체(maneuver 균형 val) · T_max 20-25 · p=0 dropout 실험 · replay rollout 평가

자세한 분석: `Sources/Research/Map-Nav Training Validity Analysis 2026-06-12.md` (WIP)
결정 로그: `Wiki/Projects/Dynav/Map-Nav.md` §2026-06-12 (DainOS vault)
