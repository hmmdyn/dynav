# dynav — Map Navigation Model

경량 듀얼모드 야외 로봇 내비게이션 모델. OSM 지도 이미지와 자기중심 카메라 관측을 결합해 장거리(500m–1km) 경로 추종을 위한 상대 웨이포인트를 예측합니다.

**로봇 플랫폼:** Clearpath Jackal UGV (4WD 차동 구동, `cmd_vel` 인터페이스)  
**타겟 디바이스:** NVIDIA Jetson Orin Nano Super (8GB RAM, 67 TOPS)

---

## 모델 구조

```
관측 이미지 (4장)               지도+경로 이미지 (1장)
전방 현재 + 과거 3장             OSM 타일 + 경로 오버레이 (heading-up)
        │                               │
  VisualEncoder                   MapEncoder
  EfficientNet-B0                 EfficientNet-B0
  GAP → Linear(1280, 256)         GAP → Linear(1280, 256)
  (B, 4, 256)                     (B, 1, 256)
        │                               │
        └──────────── Decoder ──────────┘
             SelfAttention (기본)
             [obs×4, map×1] → 5-token Transformer × 4
             → obs 위치 mean pool → (B, 256)
             or CrossAttention (ablation)
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
- 타일: CartoDB Voyager nolabels, zoom=19
- render_size=512, output_size=224, **crop_ratio=0.7** (중앙 70% 크롭 후 업스케일 → 1.43× 줌)
- heading-up 고정 (로봇 heading이 항상 위를 향함)
- 로봇 마커: 파란 원 (r=6 px), 목적지: 초록 원 (r=5 px)
- 경로: 현재 위치 → 목적지 구간만 빨간 선 (과거 구간 미표시)

**경로 생성:**

| 컨텍스트 | 방법 | 특징 |
|----------|------|------|
| FrodoBots 학습 | `osm_snap.py` — Overpass API + Dijkstra | 오프라인, 보행자 전용 네트워크 |
| rosbag 학습 | GPS 궤적 → OSRM `/match` | EKF GPS, 에피소드당 1회 |
| 추론 (ROS) | 현재+목적지 → OSRM `/route` | 미션 시작 시 1회 호출 후 캐시 |

**지도 모드:**  
`"rgb"` 단일 모드 사용. `"hybrid"` 모드는 구현 완료이나 현 연구에서 미사용 (후속 연구 보류).

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
python scripts/train.py data.data_dir=/path/to/data
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
# 전체 데이터셋 빌드 (configs/paths.yaml에서 경로 설정)
python scripts/build_frodobots_dataset.py

# GUI에서 실행 (파라미터 조정 + 로그 + 샘플 리뷰)
python scripts/dynav_gui.py

# 생성된 샘플 확인
python scripts/view_samples.py
```

내부 동작:
1. `valid_segments_*.json` 로드 (사전 필터링된 세그먼트)
2. 세그먼트당 `fetch_ped_network()` → 보행자 네트워크 다운로드 (캐시)
3. `snap_trajectory()` → 품질 필터 (GPS↔OSM 평균 거리 < 10m)
4. `snap_trajectory_graph()` → Dijkstra 코너 보정 경로 생성
5. 샘플마다 `MapRenderer.render()` → map.png
6. 세그먼트 전체 GIF 생성 + manifest.json
7. `ride_id % 10 == 0` → val, 나머지 → train

**필터 파라미터 (env var 또는 GUI에서 조정):**

| Env Var | 기본값 | 설명 |
|---------|--------|------|
| `DYNAV_OSM_SNAP_THRESH` | 10.0 m | GPS↔OSM 평균 거리 상한 |
| `DYNAV_NET_DISP` | 10.0 m | 세그먼트 순 변위 하한 |
| `DYNAV_SPEED` | 0.7 m/s | 평균 속도 하한 |
| `DYNAV_STRAIGHTNESS` | 0.75 | 순변위/누적거리 하한 |
| `DYNAV_SAMPLE_STRIDE` | 20 frames | 샘플 간격 (1s @20fps) |

### 데이터셋 구조

```
dataset/
  train/
    sample_{ride_id}_{frame_id}/
      obs_0.png        # 전방 카메라, 현재 프레임
      obs_1.png        # 전방 카메라, 0.5초 전
      obs_2.png        # 전방 카메라, 1.0초 전
      obs_3.png        # 전방 카메라, 1.5초 전
      map.png          # OSM 지도 + 경로 오버레이 (224×224, heading-up, crop_ratio=0.7)
      meta.json        # gt_waypoints, route_direction, osm_snap_mean_m, …
    ...
  val/
  gifs/
    ride_{id}_seg_{idx}.gif   # 세그먼트 전체 heading-up 애니메이션
    manifest.json             # gif → 샘플 목록 + 필터 통계 매핑
  frames/
    ride_{id}/
      {frame_id:06d}.jpg      # 사전 추출된 프레임 (OBS_STRIDE=10 간격)
```

---

## 학습

```bash
# 기본
python scripts/train.py

# 설정 오버라이드
python scripts/train.py \
    training.batch_size=64 \
    decoder.type=cross_attention \
    training.learning_rate=3e-4

# Hybrid 지도 ablation
python scripts/train.py --config-name ablation_map_hybrid
```

---

## 설정 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `map.tile_zoom` | 19 | OSM 타일 줌 레벨 |
| `map.render_size` | 512 | 회전 전 캔버스 크기 (px) |
| `map.output_size` | 224 | 최종 지도 이미지 크기 (px) |
| `map.crop_ratio` | 0.7 | 회전 후 중앙 크롭 비율 (< 1.0 이면 줌인) |
| `map.mode` | `"rgb"` | `"hybrid"` 구현됨, 현재 미사용 |
| `model.token_dim` | 256 | 임베딩 차원 |
| `model.obs_context_length` | 3 | 과거 전방 프레임 수 (총 obs = K+1) |
| `model.prediction_horizon` | 5 | 예측 웨이포인트 수 |
| `decoder.type` | `self_attention` | `cross_attention`으로 변경 시 ablation |
| `encoder.freeze_epochs` | 5 | 초기 backbone 고정 에폭 수 |
| `data.max_waypoint_distance` | 2.5 | 웨이포인트 정규화 기준 (미터) |

---

## 파일 구조

```
dynav/
├── configs/
│   ├── map.yaml                      # 지도 공용 설정 (tile, render, crop_ratio)
│   ├── paths.yaml                    # 환경별 데이터 경로
│   ├── default.yaml                  # 전체 하이퍼파라미터
│   ├── ablation_*.yaml               # ablation 설정
│   ├── rosbag_topics.yaml
│   └── record_topics.yaml
├── dynav/
│   ├── map/                          # 통합 지도 모듈
│   │   ├── tiles.py                  # TileCache, stitch_tiles
│   │   ├── routing.py                # OSRMRouter (추론/rosbag용)
│   │   ├── osm_snap.py               # Overpass+Dijkstra 오프라인 스냅 (FrodoBots용)
│   │   ├── segment.py                # segment_gps_episode
│   │   ├── overlay.py                # 시각화 상수 + rgb/hybrid 드로잉
│   │   └── renderer.py               # MapRenderer (crop_ratio 지원)
│   ├── models/ (encoders, decoders, map_nav_model.py)
│   ├── data/ (dataset.py, transforms.py)
│   ├── losses/ (navigation_losses.py)
│   └── utils/ (geometry.py, visualization.py)
├── scripts/
│   ├── train.py
│   ├── extract_rosbag.py             # rosbag → 학습 샘플 (OSRM)
│   ├── build_frodobots_dataset.py    # FrodoBots → 학습 샘플 (osm_snap)
│   ├── dynav_gui.py                  # 파이프라인 GUI
│   ├── view_samples.py               # 샘플 빠른 뷰어
│   ├── record_bag.py
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
