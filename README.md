# dynav — Map Navigation Model

경량 듀얼모드 야외 로봇 내비게이션 모델. OSM 지도 이미지와 자기중심 카메라 관측을 결합해 장거리(500m–1km) 경로 추종을 위한 상대 웨이포인트를 예측합니다.

**로봇 플랫폼:** Clearpath Jackal UGV (4WD 차동 구동, `cmd_vel` 인터페이스)  
**타겟 디바이스:** NVIDIA Jetson Orin Nano Super (8GB RAM, 67 TOPS)

---

## 모델 구조

```
관측 이미지 (4장)          지도+경로 이미지 (1장)
전방 현재/과거 2장 + 후방    OSM 타일 + 경로 오버레이 (heading-up)
        │                          │
  VisualEncoder              MapEncoder
  EfficientNet-B0            EfficientNet-B0
  GAP → Linear(1280, 256)    3×3 adaptive pool → Linear(1280, 256)
  (B, 4, 256)                (B, 9, 256)
        │                          │
        └──────── Decoder ─────────┘
              CrossAttention (기본)
              or SelfAttention (ViNT 스타일, ablation용)
                      │
                 (B, 256)
                      │
               WaypointHead
          Linear(256,128) → ReLU → Linear(128, H×2)
                      │
             웨이포인트 (B, 5, 2)
           로봇 바디 프레임 상대 좌표, [-1, 1] 정규화
```

---

## 지도 모듈 (`dynav/map/`)

모든 파이프라인(rosbag, FrodoBots, 추론)이 동일한 `MapRenderer`를 사용합니다.

**공통 렌더링 스펙:**
- zoom=19, render_size=512, output_size=224
- heading-up 고정 (로봇 heading이 항상 위를 향함)
- 로봇 마커: 파란 원 (r=6 px) — heading-up에서 화살표는 불필요
- 목적지 마커: 초록 원 (r=5 px)
- 경로: 현재 위치 → 목적지 구간만 빨간 선 (과거 구간 미표시)

**경로 생성:**

| 컨텍스트 | 방법 | 특징 |
|----------|------|------|
| rosbag 학습 | GPS 궤적 → OSRM `/match` | 에피소드당 1회 호출 |
| FrodoBots 학습 | 세그먼트 GPS → OSRM `/match` (radius=50m) | 세그먼트당 1회 호출 |
| 추론 | 현재+목적지 → OSRM `/route` | 미션 시작 시 1회 호출 후 캐시 |

두 경우 모두 OSM 네트워크 위 경로 → 학습/추론 분포 일치.

**두 가지 지도 모드 (`ablation_map_hybrid.yaml`으로 전환):**

| 모드 | Ch R | Ch G | Ch B |
|------|------|------|------|
| `"rgb"` (기본) | OSM 타일 R | OSM 타일 G | OSM 타일 B |
| `"hybrid"` | OSM 그레이스케일 | Gaussian 경로 마스크 | 로봇(amp=1.0) + 목적지(amp=0.5) Gaussian |

---

## 설치

```bash
pip install torch torchvision einops hydra-core omegaconf wandb
pip install requests pillow rosbags opencv-python tqdm
```

서브모듈 없음 — `dynav/map/` 패키지가 모든 지도 기능을 포함합니다.

---

## 빠른 시작

```bash
# 파이프라인 검증 (실제 데이터 불필요)
python scripts/train.py training.dummy=true

# Sanity check (100 iter 후 waypoint_loss < 0.05)
python scripts/sanity_check.py
```

---

## 데이터 파이프라인

### rosbag (Insta360 + Jackal J100)

```bash
# 1. 녹화
python scripts/record_bag.py --output ~/bags/run_01

# 2. 학습 샘플 추출
python scripts/extract_rosbag.py \
    --bag ~/bags/run_01/ \
    --output data/ \
    --split train \
    --goal-lat 37.562 --goal-lon 126.941
```

내부 동작:
1. OSRM `/match`로 GPS 궤적을 OSM 도로망에 스냅 (에피소드당 1회)
2. 평균 편차 > 10m인 에피소드 스킵
3. 프레임마다 `MapRenderer.render()` → `map.png` 생성
4. `route_direction` = `compute_route_direction()` (라디안, 바디 프레임)

### FrodoBots

```bash
python scripts/build_frodobots_dataset.py \
    --frodo-root /path/to/frodobots \
    --output data/
```

내부 동작:
1. `segment_gps_episode()`로 에피소드 → 클린 세그먼트 분할:
   - GPS 점프(>5m) 지점에서 분리
   - 장기 정지 구간 제거 (< 0.3m/s가 5초 이상)
   - 시작점 복귀(루프) 시 세그먼트 종료
   - 총 경로 < 10m 세그먼트 폐기
2. 세그먼트당 OSRM `/match` (radius=50m, 1Hz GPS 대응)
3. `ride_id % 10 == 0` → val, 나머지 → train

### 데이터셋 구조

```
data/
  train/
    sample_000000/
      obs_0.png        # 전방 카메라, 현재 프레임
      obs_1.png        # 전방 카메라, 0.5초 전
      obs_2.png        # 전방 카메라, 1.0초 전
      obs_3.png        # 후방 카메라, 현재 프레임
      map.png          # OSM 지도 + 경로 오버레이 (224×224, heading-up)
      meta.json        # gt_waypoints, route_direction
    ...
  val/
```

---

## 학습

```bash
# 기본
python scripts/train.py

# 설정 오버라이드
python scripts/train.py \
    training.batch_size=64 \
    decoder.type=self_attention \
    training.learning_rate=3e-4

# Hybrid 지도 ablation
python scripts/train.py --config-name ablation_map_hybrid
```

---

## 설정 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `map.tile_zoom` | 19 | OSM 타일 줌 레벨 |
| `map.mode` | `"rgb"` | `"hybrid"`으로 변경 시 채널 분리 모드 |
| `map.render_size` | 512 | 회전 전 캔버스 크기 (px) |
| `map.output_size` | 224 | 최종 지도 이미지 크기 (px) |
| `model.token_dim` | 256 | 임베딩 차원 |
| `model.prediction_horizon` | 5 | 예측 웨이포인트 수 |
| `decoder.type` | `cross_attention` | `self_attention`으로 변경 시 ViNT 스타일 |
| `encoder.freeze_epochs` | 5 | 초기 backbone 고정 에폭 수 |
| `data.max_waypoint_distance` | 2.5 | 웨이포인트 정규화 기준 (미터) |

---

## 파일 구조

```
dynav/
├── configs/
│   ├── map.yaml                      # 지도/OSRM 공용 설정
│   ├── default.yaml                  # 전체 하이퍼파라미터
│   ├── ablation_map_hybrid.yaml      # hybrid 모드 ablation
│   ├── rosbag_topics.yaml            # rosbag 추출 설정
│   └── record_topics.yaml            # rosbag 녹화 토픽
├── dynav/
│   ├── map/                          # 통합 지도 모듈
│   │   ├── tiles.py                  # TileCache, stitch_tiles
│   │   ├── routing.py                # OSRMRouter, is_route_valid, find_current_idx
│   │   ├── segment.py                # segment_gps_episode (FrodoBots)
│   │   ├── overlay.py                # 시각화 상수 + rgb/hybrid 드로잉
│   │   └── renderer.py               # MapRenderer
│   ├── models/ (encoders, decoders, map_nav_model.py)
│   ├── data/ (dataset.py, transforms.py)
│   ├── losses/ (navigation_losses.py)
│   └── utils/ (geometry.py, visualization.py)
├── scripts/
│   ├── train.py
│   ├── extract_rosbag.py             # rosbag → 학습 샘플
│   ├── build_frodobots_dataset.py    # FrodoBots → 학습 샘플
│   ├── record_bag.py
│   ├── sanity_check.py
│   └── visualize_attention.py
└── tests/
    ├── test_map.py
    ├── test_encoders.py, test_decoders.py, test_model.py, test_losses.py
```

---

## 관련 연구

- [ViNT](https://general-navigation-models.github.io/vint/) — Visual Navigation Transformer
- [NoMaD](https://general-navigation-models.github.io/nomad/) — No Map Diffusion
- [LeLaN](https://lelan.github.io/) — Language-conditioned Navigation
