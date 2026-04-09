# dynav 학습 데이터 수집 가이드

Insta360 X2/X3 카메라와 Clearpath Jackal J100을 사용하여 Map Navigation Model (dynav) 학습 데이터를 수집하고 준비하는 방법을 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [환경 요구사항](#2-환경-요구사항)
3. [데이터 수집](#3-데이터-수집)
4. [데이터셋 구조](#4-데이터셋-구조)
5. [meta.json 스키마](#5-metajson-스키마)
6. [데이터 품질 검증](#6-데이터-품질-검증)
7. [권장 데이터셋 구성](#7-권장-데이터셋-구성)
8. [알려진 한계 및 향후 작업](#8-알려진-한계-및-향후-작업)

---

## 1. 개요

```
Jackal J100 주행 → rosbag 녹화 (record_bag.py)
        ↓
   rosbag 후처리 (extract_rosbag.py)
        ↓
  map.png + obs_*.png + meta.json  (DyNavDataset 포맷)
```

**하드웨어:**
- 카메라: Insta360 X2/X3 (`insta360_ros_driver`)
- 로봇: Clearpath Jackal J100 (`j100_0519`)

파이프라인 검증만 필요한 경우 실제 데이터 없이 `DummyDyNavDataset`을 사용할 수 있습니다:

```bash
python scripts/sanity_check.py
python scripts/train.py training.dummy=true
```

---

## 2. 환경 요구사항

### 필수 Python 패키지

```bash
pip install rosbags Pillow omegaconf tqdm opencv-python
```

| 패키지 | 용도 |
|--------|------|
| `rosbags` | ROS2 `.db3` 파일 읽기 (ROS2 설치 불필요) |
| `Pillow` | 이미지 저장/변환 |
| `omegaconf` | YAML 설정 파일 로딩 |
| `tqdm` | 진행률 표시 |
| `opencv-python` | 압축 이미지 디코딩 |

### osmnav 서브모듈

```bash
git submodule update --init --recursive

# 정상 확인
ls osmnav/src/osmnav/
ls osmnav/src/nomad_map_context/
```

### 선택적 요구사항

- **ROS2 Humble**: rosbag 녹화 시 필요. `rosbags` 패키지로 `.db3` 파일 읽기는 ROS2 없이도 가능.
- **인터넷 접속**: OSM 타일 서버 접속 필요 (후처리 시). 오프라인이라면 타일을 사전 캐시하거나 자체 타일 서버를 구성.

---

## 3. 데이터 수집

### 3.1 로봇 설정 확인

녹화 전에 필요한 토픽이 발행되고 있는지 확인합니다.

```bash
# Insta360 카메라 프레임 레이트 확인
ros2 topic hz /fisheye/dual/image/compressed

# J100 GPS 수신 확인 (status.status >= 0 이면 FIX 상태)
ros2 topic echo /j100_0519/sensors/gps_0/fix --once

# J100 GPS 헤딩 확인
ros2 topic echo /j100_0519/sensors/gps_0/heading --once

# Insta360 IMU 확인
ros2 topic echo /imu/data --once
```

**토픽 대응표:**

| 데이터 | 토픽 | 타입 |
|--------|------|------|
| 듀얼 카메라 (권장) | `/fisheye/dual/image/compressed` | `sensor_msgs/CompressedImage` |
| 전방 카메라 | `/fisheye/front/image` | `sensor_msgs/Image` |
| 후방 카메라 | `/fisheye/back/image` | `sensor_msgs/Image` |
| GPS fix | `/j100_0519/sensors/gps_0/fix` | `sensor_msgs/NavSatFix` |
| GPS 헤딩 | `/j100_0519/sensors/gps_0/heading` | `geometry_msgs/QuaternionStamped` |
| IMU (filtered) | `/imu/data` | `sensor_msgs/Imu` |

> **권장:** 프레임 드롭 방지를 위해 `/fisheye/dual/image/compressed`를 녹화하고 후처리에서 전방/후방으로 분리.

### 3.2 주행 가이드라인

**경로 선택:**
- 최소 경로 길이: **200m** (더 짧으면 샘플 수 부족)
- 권장 경로 길이: 500m ~ 1km
- 피해야 할 환경: 터널/지하 (GPS 손실), 고층 건물 밀집지 (GPS 다중경로 오류), 밀림/숲 (수목 GPS 차단)

**주행 방법:**
- 권장 속도: **0.5 ~ 1.5 m/s**
  - 너무 빠르면 카메라 모션 블러, GPS 샘플 밀도 저하
  - 너무 느리면 정지 프레임 필터(`min_speed_mps: 0.3`)에 의해 샘플 제거
- 출발 전 GPS FIX 대기: `status.status >= 0` 확인 후 출발
- **같은 경로를 3회 이상 주행** (다른 시간대: 오전/오후/저녁) — 조명 변화 다양성 확보

### 3.3 rosbag 녹화

`scripts/record_bag.py`를 사용합니다. 녹화할 토픽은 `configs/record_topics.yaml`로 관리됩니다.

```bash
# 기본 녹화 (Ctrl-C로 종료)
python scripts/record_bag.py

# 시간 제한 녹화 (120초)
python scripts/record_bag.py --duration 120

# 출력 디렉토리 지정
python scripts/record_bag.py --output ~/bags/campus_run_01
```

기본 녹화 토픽 (`configs/record_topics.yaml`):
- `/fisheye/dual/image/compressed`
- `/j100_0519/sensors/gps_0/fix`
- `/j100_0519/sensors/gps_0/heading`
- `/imu/data_raw`

**저장 용량 예상:**

| 설정 | 용량 |
|------|------|
| dual compressed (zstd), 30Hz | ~25 MB/min |
| 10분 주행 | ~250 MB |

### 3.4 rosbag 확인

```bash
ros2 bag info ~/bags/campus_run_01/
```

출력 예시:

```
Duration:          612.345s
Topic information: Topic: /fisheye/dual/image/compressed | Count: 18369
                   Topic: /j100_0519/sensors/gps_0/fix   | Count: 612
                   Topic: /j100_0519/sensors/gps_0/heading | Count: 612
                   Topic: /imu/data_raw                  | Count: 61230
```

확인 항목:
- 카메라 메시지 수 ≈ 녹화 시간(초) × 30 (Insta360 30Hz)
- GPS 메시지 수 ≈ 녹화 시간(초) × 1 (1Hz)
- `gps_0/heading` 메시지 수 ≈ `gps_0/fix`와 동일

### 3.5 데이터 추출

rosbag에서 학습 샘플을 추출합니다.

```bash
# 기본 (목적지 = bag의 마지막 GPS 위치)
python scripts/extract_rosbag.py \
    --bag ~/bags/campus_run_01/ \
    --output data/ \
    --split train

# 목적지 좌표 지정 (권장)
python scripts/extract_rosbag.py \
    --bag ~/bags/campus_run_01/ \
    --output data/ \
    --split train \
    --goal-lat 37.562 \
    --goal-lon 126.941
```

목적지를 지정하면 지도 이미지에 초록색 마커가 렌더링됩니다. 미지정 시 bag의 마지막 GPS 위치가 목적지로 사용됩니다.

**설정 커스터마이즈:**

```bash
python scripts/extract_rosbag.py \
    --bag ~/bags/campus_run_01/ \
    --output data/ \
    --split train \
    --config configs/rosbag_topics.yaml
```

**ROS1 bag 처리:**

```bash
python scripts/extract_rosbag.py \
    --bag old_recording.bag \
    --output data/ \
    --split train \
    --ros-version 1
```

### 3.6 토픽명 불일치 해결

bag에서 실제 토픽명 확인:

```bash
ros2 bag info ~/bags/campus_run_01/
# 또는
python -c "
from rosbags.rosbag2 import Reader
with Reader('~/bags/campus_run_01/') as r:
    for c in r.connections: print(c.topic, c.msgtype)
"
```

`configs/rosbag_topics.yaml` 수정 예시:

듀얼 스트림으로 녹화한 경우:
```yaml
topics:
  dual_camera: "/fisheye/dual/image/compressed"
  gps:         "/j100_0519/sensors/gps_0/fix"
  gps_heading: "/j100_0519/sensors/gps_0/heading"
  imu:         "/imu/data"
```

전방/후방 분리 녹화한 경우:
```yaml
topics:
  front_camera: "/fisheye/front/image"
  rear_camera:  "/fisheye/back/image"
  gps:          "/j100_0519/sensors/gps_0/fix"
  gps_heading:  "/j100_0519/sensors/gps_0/heading"
  imu:          "/imu/data"
```

**자주 발생하는 토픽명 불일치:**

| 문제 상황 | 기본 설정 | 실제 토픽 |
|-----------|-----------|-----------|
| 다른 로봇 네임스페이스 | `/j100_0519/sensors/gps_0/fix` | `/j100_XXXX/sensors/gps_0/fix` |
| GPS 헤딩 없음 (fallback) | `heading_source: gps_heading` | `heading_source: imu` 또는 `gps` |
| RealSense 카메라 | `/fisheye/front/image` | `/camera/color/image_raw` |
| Microstrain IMU | `/imu/data` | `/microstrain/imu/data` |

---

## 4. 데이터셋 구조

`extract_rosbag.py` 실행 후 생성되는 디렉토리 구조입니다.

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
│   └── sample_XXXXXX/
├── val/
└── test/
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

---

## 5. meta.json 스키마

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

**`route_direction`** — `float`, 라디안, 바디 프레임

- 다음 GPS 궤적 방향각 (바디 프레임 기준)
- 0.0 = 직진, +π/2 = 왼쪽 90도, -π/2 = 오른쪽 90도
- 손실 함수 `L_direction`에서 사용

**`robot_lat`, `robot_lon`** — 디버그용, 모델 학습에 미사용

**`heading_deg`** — 나침반 각도 (0=북쪽, 90=동쪽, CW)

**`gt_latlon`** — 정답 웨이포인트의 원래 GPS 좌표, 디버그/시각화용

---

## 6. 데이터 품질 검증

### 6.1 파이프라인 검증

실제 데이터 없이 학습 파이프라인이 올바르게 동작하는지 먼저 확인합니다.

```bash
python scripts/sanity_check.py
# 정상: 100 iter 후 waypoint_loss < 0.05, exit code 0

python scripts/train.py training.dummy=true training.dummy_size=50
```

### 6.2 데이터셋 로딩 테스트

```python
from dynav.data.dataset import DyNavDataset

ds = DyNavDataset("data/", split="train")
print(f"샘플 수: {len(ds)}")

sample = ds[0]
print("observations shape:", sample["observations"].shape)   # (4, 3, 224, 224)
print("map_image shape:", sample["map_image"].shape)          # (3, 224, 224)
print("gt_waypoints shape:", sample["gt_waypoints"].shape)   # (5, 2)
print("gt_waypoints:", sample["gt_waypoints"])
print("route_direction:", sample["route_direction"])
```

### 6.3 웨이포인트 정상성 확인

**직진 주행 시 예상 패턴:**

- `gt_waypoints[:, 0]` (x, 전방) > 0 — 항상 양수여야 함
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

### 6.4 지도 이미지 확인

```python
from PIL import Image

map_img = Image.open("data/train/sample_000000/map.png")
map_img.show()
```

정상적인 지도 이미지:
- 파란 화살표: 이미지 하단 1/3 부근 (로봇 위치)
- 빨간 선: 이미지 상단 방향으로 뻗어 있음 (전방 경로)
- 초록 마커: 목적지 위치

---

## 7. 권장 데이터셋 구성

### 7.1 최소 구성

| 항목 | 최솟값 | 권장값 |
|------|--------|--------|
| 경로 수 | 2개 | 5개 이상 |
| 경로당 주행 횟수 | 1회 | 3회 (다른 시간대) |
| 총 경로 길이 | 500m | 3km 이상 |
| 학습 샘플 수 | 500개 | 3,000개 이상 |

### 7.2 Train/Val/Test 분할

데이터를 **경로 단위로** 분할합니다. 같은 경로의 다른 타임스탬프 샘플이 train/val에 동시에 들어가면 리크가 발생합니다.

```bash
# 경로 A, B, C → train
python scripts/extract_rosbag.py --bag bag_route_A/ --output data/ --split train
python scripts/extract_rosbag.py --bag bag_route_B/ --output data/ --split train
python scripts/extract_rosbag.py --bag bag_route_C/ --output data/ --split train

# 경로 D → val (새로운 경로)
python scripts/extract_rosbag.py --bag bag_route_D/ --output data/ --split val

# 경로 E → test (새로운 경로)
python scripts/extract_rosbag.py --bag bag_route_E/ --output data/ --split test
```

권장 분할 비율: **Train 70% / Val 15% / Test 15%**

### 7.3 디렉토리 명명 규칙

```
data/
└── real_{장소}_{날짜}/
    ├── train/
    ├── val/
    └── test/

예시:
  real_kaist_campus_20260409/
  real_hanyang_20260415/
```

---

## 8. 알려진 한계 및 향후 작업

### 8.1 현재 한계

**GPS 정확도:**
일반 GNSS GPS의 위치 오차는 약 3 ~ 5m입니다. 웨이포인트 정규화 기준이 2.5m이므로 GPS 오차가 정답 레이블 품질에 직접 영향을 줍니다. 높은 정밀도가 필요하다면 RTK GPS(오차 < 0.1m)를 권장합니다.

**IMU-카메라 외부 캘리브레이션 미적용:**
GPS 헤딩(`gps_0/heading`)을 직접 사용하며, 카메라와 GPS 안테나 간의 외부 파라미터를 보정하지 않습니다. 안테나가 로봇 전방 중심에 장착된 경우 오차가 작지만, 오프셋 마운팅이 있다면 캘리브레이션이 필요합니다.

**타임스탬프 동기화:**
카메라 30Hz, GPS 1Hz를 최근접값(nearest-neighbor)으로 매칭합니다. `gps_tol_s: 0.5` 허용 오차 내에 GPS 메시지가 없으면 해당 프레임을 제거합니다. GPS 재밍 상황에서 샘플 드롭이 많아질 수 있습니다.

**fisheye 왜곡 미보정:**
Insta360 fisheye 이미지를 그대로 obs 이미지로 사용합니다. 원근 투영이 아니므로 시각 인코더가 왜곡된 영상에서 특징을 학습합니다. 학습과 추론 시 동일한 왜곡이 적용되므로 일관성은 있지만, 사전 학습된 EfficientNet-B0 특징과 분포 차이가 생깁니다.

### 8.2 향후 작업

| 항목 | 설명 |
|------|------|
| RTK GPS 지원 | Piksi Multi, u-blox F9P RTK NMEA 메시지 파싱 추가 |
| fisheye 언디스토션 | 원근 투영 변환 후 obs 이미지 생성 옵션 |
| 데이터 증강 | 색상 지터링, 랜덤 크롭, GPS 노이즈 주입 등 `transforms.py` 확장 |
| 자동 품질 필터 | GPS covariance 임계치, 카메라 블러 감지, 저속 구간 자동 제거 |
| 멀티세션 정렬 | 동일 경로 다회 주행 데이터의 GPS 궤적 정렬 |
