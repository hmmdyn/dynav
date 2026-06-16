# 2026-06-11 실험 사이클

**주제**: full / map-only / obs-only 비교 — 기준 학습 (3개 설정)
**환경**: 5090 머신 (git 클론 아님, zip 사본)
**설정**: map_dropout_p=0.25, map_tokens=1(GAP), batch=32, epochs=50(T_max), cosine LR 1e-3

## Runs

| run id   | 설정         | 결과                                                       | 체크포인트                                                      |
| -------- | ---------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| jk0u5d4i | —          | 즉시 종료 (크래시)                                              | 없음                                                         |
| dgnfxbhw | —          | 즉시 종료 (크래시)                                              | 없음                                                         |
| zk7cqqj2 | full#1     | 비정상 (env 문제 추정)                                          | 없음                                                         |
| wiizgrmh | map-only   | ADE 0.637 / turn 0.803/0.828 / uturn 0.872               | **유실** (저장 경로 버그)                                          |
| b83cpx5e | full#?     | —                                                        | **유실**                                                     |
| 9c2aqmej | full#2     | best ep6: ADE 0.618 / turn 0.781/0.788 / **uturn 0.889** | ep14만 생존 (ADE 0.642)                                       |
| pkx580d9 | obs-only#2 | (컨트롤)                                                    | **생존** → `checkpoints_recovered/obs_only_pkx580d9_best.pt` |

## 이슈

- **체크포인트 저장 경로 버그**: `train.py`가 `Path("checkpoints")`(CWD 상대)로 저장 → hydra가 cwd를 `outputs/<date>/<time>/`으로 바꾸지 않아 repo 루트에 저장, 연속 런이 서로 덮어씀
- 06-12에 `train.py` 수정: `HydraConfig.runtime.output_dir/checkpoints`로 저장 경로 변경 (커밋: 이 PR)
- 06-11 작업 폴더 휴지통 이동 후 전수 조사 → pkx580d9 best 생존 확인

## 복구 체크포인트

`checkpoints_recovered/` (로컬, gitignore):
- `obs_only_pkx580d9_best.pt` — val_loss WandB min과 일치 확인
