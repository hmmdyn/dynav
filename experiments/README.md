# dynav 실험 로그

실험 결과를 추적하는 디렉토리. 각 실험 사이클은 `YYYY-MM-DD/` 폴더로 관리.

## 디렉토리 구조

```
experiments/
├── README.md           ← 이 파일 (실험 색인)
└── YYYY-MM-DD/         ← 날짜별 실험 사이클
    └── runs.md         ← 해당 날짜 run 요약 + 결론
```

## 연결

- WandB: https://wandb.ai/hmmdyn/dynav
- 결과 파일: `results/YYYY-MM-DD/` (CSV·PNG)
- 체크포인트: `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/` (gitignore, 로컬)
- 결정 로그: `Wiki/Projects/Dynav/Map-Nav.md` (DainOS vault)

## 실험 사이클 색인

| 날짜 | 주제 | 결론 | 파일 |
|------|------|------|------|
| 2026-06-11 | full/map-only/obs-only 비교 기준 학습 | 체크포인트 유실(저장 경로 버그) — 06-12 복구 | [runs](2026-06-11/runs.md) |
| 2026-06-12 | GPS 노이즈 곡선 + mapdrop sweep + tokens ablation | H3 입증·mapdrop=0.1 채택·GAP 유지 | [runs](2026-06-12/runs.md) |
