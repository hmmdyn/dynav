# dynav 학습 데이터 수집 가이드

Map Navigation Model (dynav) 학습에 필요한 데이터를 수집하고 준비하는 방법을 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [환경 요구사항](#2-환경-요구사항)
3. [Stage 1: API 기반 시뮬레이션 데이터 수집](#3-stage-1-api-기반-시뮬레이션-데이터-수집)
4. [Stage 2: 실제 로봇 데이터 수집](#4-stage-2-실제-로봇-데이터-수집)
5. [데이터셋 구조](#5-데이터셋-구조)
6. [meta.json 스키마](#6-metajson-스키마)
7. [데이터 품질 검증](#7-데이터-품질-검증)
8. [권장 데이터셋 구성](#8-권장-데이터셋-구성)
9. [알려진 한계 및 향후 작업](#9-알려진-한계-및-향후-작업)

---

## 1. 개요

dynav 학습 데이터 파이프라인은 두 단계로 구성됩니다.

```
Stage 1: 시뮬레이션 데이터 (로봇 불필요)
  OSRM routing API → 경로 좌표 → OSM 타일 렌더링 → meta.json 생성
  스크립트: scripts/collect_data.py
  결과: map.png (실제) + obs_*.png (회색 플레이스홀더)

Stage 2: 실제 로봇 데이터
  Jackal UGV 주행 → rosbag 녹화 → 프레임 추출 + 지도 렌더링
  스크립트: scripts/extract_rosbag.py
  결과: map.png (실제) + obs_*.png (실제 카메라 이미지)
```

**Stage 1**은 인터넷 접속만 있으면 실행할 수 있어 초기 학습 파이프라인 검증에 적합합니다. 카메라 이미지가 회색 플레이스홀더이므로 모델이 맵 정보만으로 학습됩니다.

**Stage 2**는 실제 환경에서 카메라 이미지와 GPS를 함께 녹화하여 현실적인 데이터를 생성합니다. Stage 1로 파이프라인을 검증한 뒤 실제 로봇 주행으로 전환하는 것을 권장합니다.

---

## 2. 환경 요구사항

### 필수 Python 패키지

```bash
pip install rosbags Pillow requests omegaconf tqdm opencv-python
```

| 패키지 | 용도 |
|--------|------|
| `rosbags` | ROS2 `.db3` 파일 읽기 (ROS2 설치 불필요) |
| `Pillow` | 이미지 저장/변환 |
| `requests` | OSM 타일 HTTP 다운로드 |
| `omegaconf` | YAML 설정 파일 로딩 |
| `tqdm` | 진행률 표시 |
| `opencv-python` | 이미지 압축 해제 및 전처리 |

### osmnav 서브모듈

```bash
# 리포지토리 최초 클론 시
git submodule update --init --recursive

# 서브모듈 정상 확인
ls osmnav/src/osmnav/
ls osmnav/src/nomad_map_context/
```

### 선택적 요구사항

- **ROS2 Humble** (Ubuntu 22.04): 실제 rosbag 녹화 시 필요. `rosbags` 패키지 사용 시 ROS2 없이도 `.db3` 파일 읽기 가능.
- **인터넷 접속**: OSM 타일 서버 (`cartodb-basemaps-a.global.ssl.fastly.net`) 및 OSRM 공개 API (`router.project-osrm.org`) 접속 필요. 오프라인 환경이라면 타일을 사전에 캐시하거나 자체 OSRM 서버를 구성해야 합니다.

---

## 3. Stage 1: API 기반 시뮬레이션 데이터 수집

`scripts/collect_data.py`는 OSRM 공개 라우팅 API로 경로를 조회하고, OSM 타일을 다운로드하여 지도 이미지를 생성합니다. 카메라 이미지는 회색 플레이스홀더로 채워집니다.

### 3.1 단일 경로 수집

```bash
python scripts/collect_data.py \
    --start 37.557 126.936 \
    --end   37.562 126.941 \
    --output data/ \
    --split train \
    --step 1.0
```

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--start LAT LON` | 출발점 위도/경도 | 필수 |
| `--end LAT LON` | 도착점 위도/경도 | 필수 |
| `--output DIR` | 데이터셋 루트 디렉토리 | `data/` |
| `--split` | `train`, `val`, `test` 중 선택 | `train` |
| `--step M` | 경로 샘플링 간격 (미터) | `1.0` |
| `--tile-cache DIR` | OSM 타일 캐시 디렉토리 | `/tmp/nomad_tile_cache` |
| `--zoom N` | 타일 줌 레벨 (17 권장) | `17` |
| `--seed N` | 랜덤 시드 | `42` |

### 3.2 배치 수집 (routes.json)

여러 경로를 한 번에 수집하려면 JSON 파일을 사용합니다.

```bash
python scripts/collect_data.py \
    --routes-json routes.json \
    --output data/ \
    --split train \
    --step 1.0
```

**routes.json 형식:**

```json
[
  {
    "start": [37.557, 126.936],
    "end":   [37.562, 126.941]
  },
  {
    "start": [37.560, 126.930],
    "end":   [37.565, 126.935]
  },
  {
    "start": [37.550, 126.920],
    "end":   [37.558, 126.928]
  }
]
```

> 좌표는 `[위도(lat), 경도(lon)]` 순서입니다.

### 3.3 권장 샘플링 파라미터

| 용도 | `--step` | 예상 샘플 수 |
|------|----------|-------------|
| 학습 데이터 (train) | `1.0m` | ~1,000 샘플/km |
| 검증 데이터 (val) | `2.0m` | ~500 샘플/km |
| 테스트 데이터 (test) | `2.0m` | ~500 샘플/km |

500m 경로에서 `--step 1.0`을 사용하면 약 500개 샘플이 생성됩니다.

### 3.4 실행 예시 및 예상 출력

```
[Route 1/3]
  OSRM route (37.5570,126.9360) → (37.5620,126.9410)
  Route: 124 waypoints, 687 m
  Sampling: 687 positions at step=1.0 m
    10/687 samples written...
    20/687 samples written...
    ...
Done. 2103 samples written to data/train/
```

### 3.5 한계 사항

- **카메라 이미지가 회색 플레이스홀더**: `obs_0.png` ~ `obs_3.png`는 모두 128,128,128 RGB 균일 이미지입니다. 모델이 맵 정보만으로 학습되므로, 시각 인코더는 의미 있는 특징을 학습하지 않습니다.
- **OSRM 공개 API 속도 제한**: 대용량 배치 수집 시 요청 간격이 짧아 실패할 수 있습니다. 실패한 경로는 `[WARN] OSRM failed` 메시지로 스킵됩니다.
- **타일 캐시 활용**: 동일 지역을 반복 수집할 때는 캐시된 타일을 재사용합니다. 캐시 디렉토리를 유지하면 네트워크 요청을 크게 줄일 수 있습니다.

---

## 4. Stage 2: 실제 로봇 데이터 수집

### 4.1 로봇 설정 확인

rosbag 녹화 전에 필요한 토픽이 발행되고 있는지 확인합니다.

```bash
# 전체 토픽 목록 확인
ros2 topic list

# 카메라 프레임 레이트 확인 (10Hz 이상 권장)
ros2 topic hz /front/image_raw
ros2 topic hz /rear/image_raw

# GPS 수신 확인 (fix 상태 및 covariance 확인)
ros2 topic echo /navsat/fix --once

# IMU 수신 확인
ros2 topic echo /imu/data --once

# odometry 확인
ros2 topic echo /odometry/filtered --once
```

**Jackal UGV 주요 토픽 대응표:**

| 데이터 | 기본 토픽 | 대안 |
|--------|-----------|------|
| 전방 카메라 | `/front/image_raw` | `/camera/color/image_raw`, `/camera_front/color/image_raw` |
| 후방 카메라 | `/rear/image_raw` | `/rear_camera/image_raw`, `/camera_rear/color/image_raw` |
| GPS | `/navsat/fix` | `/gps/fix`, `/fix` |
| IMU | `/imu/data` | `/microstrain/imu/data`, `/vectornav/IMU` |
| Odometry | `/odometry/filtered` | `/odom`, `/jackal_velocity_controller/odom` |

> `/odometry/filtered`는 GPS+IMU+휠 오도메트리를 융합한 EKF 출력입니다. 가용한 경우 이를 우선 사용하고, EKF가 실행되지 않는 환경에서는 `/odom`으로 대체합니다.

### 4.2 주행 가이드라인

좋은 학습 데이터를 수집하기 위한 현장 지침입니다.

**경로 선택:**

- 최소 경로 길이: **200m** (더 짧으면 샘플 수가 부족)
- 권장 경로 길이: 500m ~ 1km
- 피해야 할 환경:
  - 터널, 지하 주차장 (GPS 신호 손실)
  - 고층 건물 밀집 지역 (GPS 다중경로 오류, covariance 폭발)
  - 밀림/숲 (수목 GPS 차단)

**주행 방법:**

- 권장 속도: **0.5 ~ 1.5 m/s** (Jackal `max_vel_x` 기본값 2.0 m/s의 절반 이하)
  - 너무 빠르면 카메라 모션 블러 발생, GPS 샘플 밀도 저하
  - 너무 느리면 정지 프레임 필터(`min_speed_mps: 0.3`)에 의해 다수 샘플 제거됨
- 녹화 시작 전 **GPS 수신 대기**: NavSatFix의 `status.status >= 0` (FIX) 확인 후 출발
- **같은 경로를 3회 이상 주행** (다른 시간대: 오전/오후/저녁)하여 조명 변화 다양성 확보

**최소 데이터셋 요건:**

- 총 경로 길이: 500m 이상
- 예상 샘플 수: ~500개 이상
- 경로 종류: 2개 이상의 서로 다른 경로

### 4.3 rosbag 녹화

```bash
ros2 bag record \
    /front/image_raw \
    /rear/image_raw \
    /navsat/fix \
    /imu/data \
    /odometry/filtered \
    -o recording_$(date +%Y%m%d_%H%M%S)
```

> 카메라를 compressed transport로 녹화하면 용량을 약 80% 절약할 수 있습니다.
>
> ```bash
> ros2 bag record \
>     /front/image_raw/compressed \
>     /rear/image_raw/compressed \
>     /navsat/fix \
>     /imu/data \
>     /odometry/filtered \
>     -o recording_$(date +%Y%m%d_%H%M%S)
> ```
>
> 이 경우 `configs/rosbag_topics.yaml`에서 `front_camera` 토픽명을 compressed 버전으로 변경하세요.

**저장 용량 예상:**

| 설정 | 용량 |
|------|------|
| raw 이미지, 10Hz | ~100 MB/min |
| compressed 이미지, 10Hz | ~20 MB/min |
| 10분 주행 (raw) | ~1 GB |
| 10분 주행 (compressed) | ~200 MB |

### 4.4 rosbag 확인

```bash
ros2 bag info recording_*.db3
```

출력 예시:

```
Files:             recording_20241215_143022.db3
Bag size:          1.2 GiB
Storage id:        sqlite3
Duration:          612.345s
Start:             Dec 15 2024 14:30:22.123 (1734226222.123)
End:               Dec 15 2024 14:40:34.468 (1734226834.468)
Messages:          73456
Topic information: Topic: /front/image_raw | Type: sensor_msgs/msg/Image | Count: 6123
                   Topic: /rear/image_raw  | Type: sensor_msgs/msg/Image | Count: 6120
                   Topic: /navsat/fix      | Type: sensor_msgs/msg/NavSatFix | Count: 612
                   Topic: /imu/data        | Type: sensor_msgs/msg/Imu | Count: 61230
                   Topic: /odometry/filtered | Type: nav_msgs/msg/Odometry | Count: 6123
```

확인 항목:
- 카메라 메시지 수 ≈ 녹화 시간(초) × 10
- GPS 메시지 수 ≈ 녹화 시간(초) × 1 (1Hz NavSatFix)
- 모든 필수 토픽이 존재하는지 확인

### 4.5 데이터 추출

rosbag에서 학습 샘플을 추출합니다.

**기본 사용법:**

```bash
python scripts/extract_rosbag.py \
    --bag recording_20241215_143022/ \
    --output data/ \
    --split train
```

**목적지 좌표 지정 (권장):**

```bash
python scripts/extract_rosbag.py \
    --bag recording_20241215_143022/ \
    --output data/ \
    --split train \
    --goal-lat 37.562 \
    --goal-lon 126.941
```

목적지를 지정하면 지도 이미지에 초록색 목적지 마커가 렌더링됩니다. 지정하지 않으면 rosbag의 마지막 GPS 위치가 목적지로 사용됩니다.

**설정 파일 커스터마이즈:**

```bash
python scripts/extract_rosbag.py \
    --bag recording_20241215_143022/ \
    --output data/ \
    --split train \
    --config configs/rosbag_topics.yaml
```

**ROS1 bag 파일 처리 (선택적):**

```bash
python scripts/extract_rosbag.py \
    --bag old_recording.bag \
    --output data/ \
    --split train \
    --ros-version 1
```

> `rosbags` 패키지는 ROS1 `.bag` 파일도 지원합니다.

### 4.6 토픽명 불일치 해결

`extract_rosbag.py` 실행 시 토픽을 찾지 못하는 오류가 발생하면:

**1. bag 파일의 실제 토픽명 확인:**

```bash
ros2 bag info recording_20241215_143022/
```

또는 rosbags Python API로 확인:

```python
from rosbags.rosbag2 import Reader
with Reader('recording_20241215_143022/') as reader:
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)
```

**2. `configs/rosbag_topics.yaml` 수정:**

```yaml
# 기본값
topics:
  front_camera: "/camera_front/color/image_raw"

# 실제 bag의 토픽명으로 변경
topics:
  front_camera: "/front/image_raw"
```

**자주 발생하는 토픽명 불일치 패턴:**

| 문제 상황 | 기본값 | 실제 토픽 |
|-----------|--------|-----------|
| RealSense 카메라 사용 | `/camera_front/color/image_raw` | `/realsense/color/image_raw` |
| 단일 카메라 설정 | `/camera_front/color/image_raw` | `/camera/color/image_raw` |
| GPSd 브릿지 | `/navsat/fix` | `/fix` |
| Jackal 기본 GPS | `/navsat/fix` | `/gps/fix` |
| Microstrain IMU | `/imu/data` | `/microstrain/imu/data` |
| EKF 없는 환경 | `/odometry/filtered` | `/odom` |

---

## 5. 데이터셋 구조

`collect_data.py` 또는 `extract_rosbag.py` 실행 후 생성되는 디렉토리 구조입니다.

```
data/
├── train/
│   ├── sample_000000/
│   │   ├── obs_0.png      # 전방 카메라, 현재 프레임 (224×224 RGB)
│   │   ├── obs_1.png      # 전방 카메라, 과거 프레임 1 (t - 0.25s)
│   │   ├── obs_2.png      # 전방 카메라, 과거 프레임 2 (t - 0.5s)
│   │   ├── obs_3.png      # 후방 카메라, 현재 프레임 (224×224 RGB)
│   │   ├── map.png        # OSM 지도 + 경로 오버레이 (224×224 RGB)
│   │   └── meta.json      # 정답 웨이포인트 및 메타데이터
│   ├── sample_000001/
│   │   └── ...
│   └── sample_XXXXXX/
│       └── ...
├── val/
│   └── sample_XXXXXX/
│       └── ...
└── test/
    └── sample_XXXXXX/
        └── ...
```

**각 파일 설명:**

| 파일 | 형식 | 설명 |
|------|------|------|
| `obs_0.png` | PNG 224×224 | 전방 카메라 현재 프레임 (t) |
| `obs_1.png` | PNG 224×224 | 전방 카메라 과거 프레임 1 (t - δ) |
| `obs_2.png` | PNG 224×224 | 전방 카메라 과거 프레임 2 (t - 2δ) |
| `obs_3.png` | PNG 224×224 | 후방 카메라 현재 프레임 (t) |
| `map.png` | PNG 224×224 | 로봇 중심 heading-up OSM 지도. 빨간선=경로, 파란 화살표=로봇 위치, 초록 마커=목적지 |
| `meta.json` | JSON | 정답 웨이포인트, 경로 방향, GPS 좌표 등 |

**실제/시뮬레이션 데이터 분리 권장 디렉토리 구조:**

```
data/
├── sim_seoul_train/        # Stage 1 시뮬레이션 데이터 (서울)
│   └── train/
├── sim_seoul_val/
│   └── val/
├── real_kaist_20241215/    # Stage 2 실제 로봇 데이터 (KAIST 캠퍼스)
│   └── train/
└── real_kaist_20241216/
    └── train/
```

---

## 6. meta.json 스키마

```json
{
  "gt_waypoints": [
    [0.42, 0.03],
    [0.51, 0.07],
    [0.48, 0.12],
    [0.39, 0.08],
    [0.44, 0.05]
  ],
  "route_direction": 0.087,
  "robot_lat": 37.557234,
  "robot_lon": 126.936512,
  "heading_deg": 45.3,
  "gt_latlon": [
    [37.557256, 126.936534],
    [37.557278, 126.936558],
    [37.557299, 126.936581],
    [37.557321, 126.936604],
    [37.557342, 126.936627]
  ]
}
```

### 필드 상세 설명

**`gt_waypoints`** — `[[dx, dy], ...]`, 길이 H=5

- 로봇 바디 프레임 기준의 정답 웨이포인트
- 좌표계: x=전방(+), y=좌측(+)
- 정규화 범위: [-1, 1]
- 정규화 기준: `max_waypoint_distance = 2.5m` (즉, x=1.0은 전방 2.5m)
- 수식: `x_norm = clip(dx_meters / 2.5, -1, 1)`

**`route_direction`** — `float`, 라디안, 바디 프레임

- 다음 3개 웨이포인트의 방향각 평균 (바디 프레임 기준)
- 0.0 = 직진, +π/2 = 왼쪽 90도, -π/2 = 오른쪽 90도
- 손실 함수 `L_direction`에서 사용

**`robot_lat`, `robot_lon`** — 디버그용

- 로봇의 WGS84 GPS 좌표
- 모델 학습에는 사용되지 않음

**`heading_deg`** — 나침반 각도 (도)

- 로봇이 향하는 방향 (0=북쪽, 90=동쪽, 시계 방향 증가)
- 바디 프레임 → 월드 프레임 변환 시 사용

**`gt_latlon`** — 디버그용

- 정답 웨이포인트의 원래 GPS 좌표 (H개)
- 시각화 및 검증에 사용

---

## 7. 데이터 품질 검증

### 7.1 파이프라인 검증 (DummydynavDataset)

실제 데이터 없이 학습 파이프라인이 올바르게 동작하는지 먼저 확인합니다.

```bash
# default.yaml에서 dummy 모드 활성화
# training.dummy: true 설정 후 실행
python scripts/train.py

# 또는 커맨드라인 오버라이드
python scripts/train.py training.dummy=true training.dummy_size=50
```

sanity_check 스크립트로 모델 전체 포워드 패스 확인:

```bash
python scripts/sanity_check.py
```

### 7.2 데이터셋 로딩 테스트

```python
from dynav.data.dataset import dynavDataset

ds = dynavDataset("data/", split="train")
print(f"샘플 수: {len(ds)}")

sample = ds[0]
print("observations shape:", sample["observations"].shape)   # (4, 3, 224, 224)
print("map_image shape:", sample["map_image"].shape)          # (3, 224, 224)
print("gt_waypoints shape:", sample["gt_waypoints"].shape)   # (5, 2)
print("gt_waypoints:", sample["gt_waypoints"])
print("route_direction:", sample["route_direction"])
```

### 7.3 웨이포인트 정상성 확인

**직진 주행 시 예상 패턴:**

- `gt_waypoints[:, 0]` (x, 전방) > 0 — 항상 양수여야 함 (앞으로 진행)
- `gt_waypoints[:, 1]` (y, 좌우) ≈ 0 — 직진 시 작은 값
- `route_direction` ≈ 0 — 직진 시 0에 가까움

```python
import json
from pathlib import Path
import statistics

data_dir = Path("data/train")
x_vals = []
route_dirs = []

for sample_dir in sorted(data_dir.iterdir())[:100]:
    with open(sample_dir / "meta.json") as f:
        meta = json.load(f)
    wps = meta["gt_waypoints"]
    x_vals.extend([wp[0] for wp in wps])
    route_dirs.append(meta["route_direction"])

print(f"웨이포인트 x 평균: {statistics.mean(x_vals):.3f}  (양수여야 정상)")
print(f"route_direction 평균: {statistics.mean(route_dirs):.3f}  (직진 경로는 0에 가까움)")
print(f"x < 0 비율: {sum(1 for x in x_vals if x < 0) / len(x_vals):.1%}  (5% 이하 권장)")
```

### 7.4 지도 이미지 확인

```python
from PIL import Image
import os

sample_dir = "data/train/sample_000000"
map_img = Image.open(f"{sample_dir}/map.png")
map_img.show()  # 파란 화살표(로봇), 빨간 선(경로), 초록 마커(목적지) 확인
```

정상적인 지도 이미지에는:
- 파란 화살표: 이미지 하단 1/3 부근에 위치 (로봇 위치)
- 빨간 선: 이미지 상단 방향으로 뻗어 있음 (목적지 방향)
- 초록 마커: 경로 끝(상단)에 위치

---

## 8. 권장 데이터셋 구성

### 8.1 최소 구성

| 항목 | 최솟값 | 권장값 |
|------|--------|--------|
| 경로 수 | 2개 | 5개 이상 |
| 경로당 주행 횟수 | 1회 | 3회 (다른 시간대) |
| 총 경로 길이 | 500m | 3km 이상 |
| 학습 샘플 수 | 500개 | 3,000개 이상 |

### 8.2 Train/Val/Test 분할

데이터를 **경로 단위로** 분할합니다. 같은 경로의 다른 타임스탬프 샘플이 train/val에 동시에 들어가면 리크가 발생합니다.

```bash
# 경로 A, B, C → train
python scripts/collect_data.py --routes-json routes_train.json \
    --output data/sim_v1 --split train --step 1.0

# 경로 D → val (새로운 경로)
python scripts/collect_data.py --routes-json routes_val.json \
    --output data/sim_v1 --split val --step 2.0

# 경로 E → test (새로운 경로)
python scripts/collect_data.py --routes-json routes_test.json \
    --output data/sim_v1 --split test --step 2.0
```

권장 분할 비율: **Train 70% / Val 15% / Test 15%**

### 8.3 시뮬레이션 + 실제 데이터 혼합

```
Phase 1 (초기 학습):
  sim 데이터 100% → 파이프라인 검증, 과적합 확인

Phase 2 (파인튜닝):
  sim 데이터 70% + real 데이터 30% → 도메인 갭 감소

Phase 3 (실제 환경 평가):
  real 데이터 100% → 최종 성능 평가
```

`data.data_dir` 설정에서 여러 데이터 경로를 지원하도록 `dynavDataset`을 확장하거나, `ConcatDataset`을 사용해 혼합합니다:

```python
from torch.utils.data import ConcatDataset
from dynav.data.dataset import dynavDataset

sim_ds  = dynavDataset("data/sim_seoul_train", split="train")
real_ds = dynavDataset("data/real_kaist_train", split="train")
mixed   = ConcatDataset([sim_ds, real_ds])
print(f"총 샘플: {len(mixed)}")
```

### 8.4 디렉토리 명명 규칙

```
data/
├── sim_{지역}_{버전}/     # 시뮬레이션 데이터
│   ├── train/
│   ├── val/
│   └── test/
└── real_{장소}_{날짜}/    # 실제 로봇 데이터
    ├── train/
    └── val/

예시:
  sim_seoul_v1/
  sim_kaist_v2/
  real_kaist_campus_20241215/
  real_hanyang_20250101/
```

---

## 9. 알려진 한계 및 향후 작업

### 9.1 현재 한계

**Stage 1 카메라 플레이스홀더:**
`collect_data.py`로 생성한 데이터의 `obs_*.png`는 모두 회색(128, 128, 128) 이미지입니다. 이 데이터로 학습한 모델은 시각 인코더가 의미 없는 특징을 학습하여, 실제 카메라 입력에 대한 일반화 성능이 낮습니다. Stage 1은 파이프라인 검증과 맵 인코더/디코더 사전 학습 용도로 사용하세요.

**GPS 정확도:**
일반 GNSS GPS의 위치 오차는 약 3 ~ 5m입니다. 웨이포인트 정규화 기준이 2.5m이므로, GPS 오차가 정답 레이블 품질에 직접 영향을 줍니다. 높은 정밀도가 필요하다면 RTK GPS(오차 < 0.1m)를 권장합니다.

**IMU-카메라 외부 캘리브레이션 미적용:**
`extract_rosbag.py`는 IMU 쿼터니언을 로봇 헤딩으로 직접 사용하며, IMU와 카메라 간의 외부 파라미터(extrinsic)를 보정하지 않습니다. 카메라가 로봇 기준 좌표계와 정렬되지 않은 경우 웨이포인트 변환에 오차가 발생합니다. Jackal의 기본 설정(카메라 전방 장착)에서는 큰 문제가 없지만, 커스텀 마운팅이 있다면 캘리브레이션이 필요합니다.

**타임스탬프 동기화:**
카메라 10Hz, GPS 1Hz의 서로 다른 레이트를 시간 근접값(nearest-neighbor)으로 매칭합니다. `gps_tol_s: 0.5` 허용 오차 내에 GPS 메시지가 없으면 해당 프레임을 제거합니다. 빠른 주행이나 GPS 재밍 상황에서 샘플 드롭이 많아질 수 있습니다.

### 9.2 향후 작업

| 항목 | 설명 |
|------|------|
| Gazebo / Isaac Sim 연동 | 포토리얼리스틱 시뮬레이션으로 Stage 1 카메라 플레이스홀더 대체 |
| RTK GPS 지원 | `extract_rosbag.py`에서 Piksi Multi, u-blox F9P RTK NMEA 메시지 파싱 추가 |
| 데이터 증강 | 카메라 색상 지터링, 랜덤 크롭, GPS 노이즈 주입 등 `transforms.py` 확장 |
| 자동 품질 필터 | GPS covariance 임계치, 카메라 블러 감지, 저속 구간 자동 제거 |
| 멀티-세션 정렬 | 여러 날 녹화한 동일 경로 데이터를 GPS 좌표 기준으로 정렬 및 병합 |

---

*최종 수정: 2026-03-18*
