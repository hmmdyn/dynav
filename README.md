# dynav — Map Navigation Model

경량 듀얼모드 야외 로봇 내비게이션 모델. OSM 지도 이미지와 자기중심 카메라 관측을 결합해 장거리(500m–1km) 경로 추종을 위한 상대 웨이포인트를 예측합니다.

**로봇 플랫폼:** Clearpath Jackal UGV (4WD 차동 구동, `cmd_vel` 인터페이스)
**타겟 디바이스:** NVIDIA Jetson Orin Nano Super (8GB RAM, 67 TOPS)

---

## 모델 구조

```
관측 이미지 (4장)          지도+경로 이미지 (1장)
전방 현재/과거 2장 + 후방    OSM 타일 + 경로 오버레이
        │                          │
  VisualEncoder              MapEncoder
  EfficientNet-B0            EfficientNet-B0
  GAP → Linear(1280, 256)    7×7 공간 유지
  (B, 4, 256)                (B, 49, 256)
        │                          │
        └──────── Decoder ─────────┘
              CrossAttention (기본)
              or SelfAttention (ViNT 스타일, ablation용)
                      │
                 (B, 256)
                      │
               WaypointHead
          Linear → ReLU → Linear → tanh
                      │
             웨이포인트 (B, 5, 2)
           로봇 바디 프레임 상대 좌표
```

### 인코더

| 인코더 | 입력 | 출력 | 특징 |
|---|---|---|---|
| VisualEncoder | (B, 4, 3, 224, 224) | (B, 4, 256) | GAP + 학습 가능한 시간적 위치 인코딩 |
| MapEncoder | (B, 3, 224, 224) | (B, 49, 256) | 7×7 공간 토큰 유지 + 2D 위치 인코딩 |

### 손실 함수

```
L_total = L_waypoint + 0.5·L_direction + 0.1·L_progress + 0.01·L_smooth
```

| 항 | 역할 |
|---|---|
| L_waypoint | L1 웨이포인트 회귀 |
| L_direction | 예측 방향 vs 경로 방향 cosine 유사도 |
| L_progress | 경로 진행 방향 전진 장려 |
| L_smooth | 궤적 부드러움 |

---

## 설치

```bash
# 의존성
pip install torch torchvision einops hydra-core omegaconf wandb
pip install requests pillow rosbags opencv-python tqdm

# 서브모듈 초기화
git submodule update --init --recursive
```

---

## 빠른 시작

### 파이프라인 검증 (데이터 불필요)

```bash
python scripts/train.py training.dummy=true
```

### Sanity check

```bash
python scripts/sanity_check.py
# 정상: 100 iter 후 waypoint_loss < 0.05, exit code 0
```

---

## 데이터 수집

### Stage 1 — API 기반 시뮬레이션 (로봇 불필요)

OSRM 경로 API와 OSM 타일로 지도 이미지를 생성하고 샘플을 만듭니다. 카메라 이미지는 placeholder입니다.

```bash
# 단일 경로
python scripts/collect_data.py \
    --start 37.557 126.936 \
    --end   37.620 126.980 \
    --output data/ --split train --step 1.0

# 여러 경로 배치
python scripts/collect_data.py \
    --routes-json routes.json \
    --output data/ --split train
```

`routes.json` 형식:
```json
[
  {"start": [37.557, 126.936], "end": [37.620, 126.980]},
  {"start": [37.500, 127.000], "end": [37.540, 127.050]}
]
```

### Stage 2 — 실제 로봇 rosbag 추출

Jackal에서 rosbag 녹화 후 실제 카메라/GPS 데이터로 샘플을 생성합니다.

```bash
# 1. rosbag 녹화 (Jackal에서)
ros2 bag record \
    /front/image_raw /rear/image_raw \
    /navsat/fix /imu/data /odometry/filtered \
    -o recording_$(date +%Y%m%d_%H%M%S)

# 2. 데이터 추출
python scripts/extract_rosbag.py \
    --bag recording_20260318_120000.db3 \
    --output data/ \
    --split train \
    --goal-lat 37.562 --goal-lon 126.941
```

토픽명이 다를 경우 `configs/rosbag_topics.yaml`을 수정하세요.
자세한 내용은 **[데이터 수집 가이드](docs/data_collection_guide.md)** 참조.

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
    sample_000001/
    ...
  val/
  test/
```

---

## 학습

```bash
# 기본 학습
python scripts/train.py

# 설정 오버라이드
python scripts/train.py \
    training.batch_size=64 \
    decoder.type=self_attention \
    training.learning_rate=3e-4
```

---

## 파일 구조

```
dynav/
├── configs/
│   ├── default.yaml              # 전체 하이퍼파라미터
│   ├── map_nav.yaml              # Map Nav 전용 설정
│   └── rosbag_topics.yaml        # rosbag 추출 설정
├── docs/
│   └── data_collection_guide.md  # 데이터 수집 가이드 (한국어)
├── dynav/
│   ├── models/
│   │   ├── encoders/             # visual_encoder.py, map_encoder.py
│   │   ├── decoders/             # cross_attention_decoder.py, self_attention_decoder.py, action_heads.py
│   │   └── map_nav_model.py      # DyNavModel (전체 조립)
│   ├── data/
│   │   ├── dataset.py            # DyNavDataset, DummyDyNavDataset
│   │   └── transforms.py         # train/eval 이미지 변환
│   ├── losses/
│   │   └── navigation_losses.py  # NavigationLoss
│   └── utils/
│       └── geometry.py           # 좌표 변환 유틸리티
├── scripts/
│   ├── train.py                  # 학습 스크립트 (Hydra)
│   ├── collect_data.py           # API 기반 데이터 수집
│   ├── extract_rosbag.py         # rosbag → 데이터셋 변환
│   ├── sanity_check.py           # CI 검증
│   └── visualize_attention.py    # Attention heatmap 시각화
├── tests/                        # 단위 테스트
└── osmnav/                       # git submodule — OSM 지도/경로 렌더링
```

---

## 설정 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `model.token_dim` | 256 | 임베딩 차원 |
| `model.prediction_horizon` | 5 | 예측 웨이포인트 수 |
| `decoder.type` | `cross_attention` | `self_attention`으로 변경 시 ViNT 스타일 |
| `decoder.n_layers` | 4 | Transformer 블록 수 |
| `encoder.freeze_epochs` | 5 | 초기 backbone 고정 에폭 수 |
| `data.max_waypoint_distance` | 2.5 | 웨이포인트 정규화 기준 (미터) |

---

## 관련 연구

- [ViNT](https://general-navigation-models.github.io/vint/) — Visual Navigation Transformer
- [NoMaD](https://general-navigation-models.github.io/nomad/) — No Map Diffusion
- [LeLaN](https://lelan.github.io/) — Language-conditioned Navigation
