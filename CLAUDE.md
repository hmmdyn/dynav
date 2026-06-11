# CLAUDE.md — Map Navigation Model (dynav)

> **이 문서는 as-built 스펙이다** — 현재 코드·`configs/`가 실제로 하는 것만 기술한다. 숫자(레이어·λ 등)의 권위는 `configs/*.yaml`이며, 본 문서는 그 값을 설명·참조하되 결정의 *이유*는 적지 않는다.
> **결정됐으나 아직 코드에 반영되지 않은 변경**(as-decided)은 여기 쓰지 않는다 — vault `Wiki/Projects/Dynav/Map-Nav.md` 결정 로그에 있고, 반영 작업은 `Daily/Tasks`로 추적된다 (루트 `CLAUDE.md` 규칙 7, 3-state).

## Project Overview

Lightweight dual-mode navigation model for outdoor robot navigation on edge devices (NVIDIA Jetson Orin Nano Super, 8GB RAM, 67 TOPS). Robot platform: Clearpath Jackal UGV (4×4 differential drive, cmd_vel interface).

**Phase 1 — 데이터 파이프라인 완성, 학습 단계 진입.** (진행 상태의 SSOT는 vault `Wiki/Projects/Dynav/Map-Nav.md` + `Daily/Projects/Map-Nav.md`.)

Two planned models:
1. **Map Navigation Model** (this phase): OSM map image + route overlay + egocentric camera → relative waypoints for long-range navigation (500m–1km).
2. **Semantic Navigation Model** (future): CLIP+FiLM language conditioning + egocentric observations for last-mile navigation.

---

## Architecture

### Inputs
- **Observations:** 4 images — front camera (current + 3 past). Each 224×224 RGB.
- **Map+Path Image:** 224×224 RGB — OSM top-down map, heading-up, route in red, robot as blue circle, goal as green circle.

### Encoders
- **VisualEncoder:** Each obs image → EfficientNet-B0 → GAP → Linear(1280, 256) → 1 token. 4 images → `obs_tokens ∈ (B, 4, 256)`.
- **MapEncoder:** Map image → EfficientNet-B0 → GlobalAvgPool → Linear(1280, 256) → 1 token → `map_tokens ∈ (B, 1, 256)`.
  - Single-token GAP is noise-robust: GPS, heading, and path centerline errors all cause horizontal displacement in heading-up maps — GAP is invariant to such shifts.

### Decoder
- **SelfAttentionDecoder** (default): Cat(obs[4], map[1]) → Self-Attn → FFN × 4. 5-token sequence. Output: mean pool over obs positions → `context ∈ (B, 256)`.
- **CrossAttentionDecoder** (ablation): Self-Attn(obs) → Cross-Attn(Q=obs, K/V=map) → FFN × 4. Output: mean pool → `context ∈ (B, 256)`.

### Action Head
`WaypointHead`: Linear(256, 128) → ReLU → Linear(128, H×2) → reshape `(H, 2)`.  
Output: H=5 relative waypoints (Δx, Δy) in robot body frame, normalized to [-1, 1].

---

## Map Module (`dynav/map/`)

Self-contained package.

### Visual constants — identical across ALL contexts
```python
ROUTE_FUTURE_COLOR = (220, 50,  50,  200)   # red: current position → goal
ROUTE_LINE_WIDTH   = 3                       # px
ROBOT_COLOR        = (30,  100, 220, 230)    # blue circle
ROBOT_RADIUS       = 6                       # px
GOAL_COLOR         = (50,  200, 80,  220)    # green circle
GOAL_RADIUS        = 5                       # px
```

Past route (already-traversed portion) is **not shown** — route displayed from current position to goal only.

### Marker spec
- **Robot**: blue filled circle, radius 6 px. Heading-up is fixed → direction arrow is redundant.
- **Goal**: green filled circle, radius 5 px.
- **Route**: red polyline, current_idx → end, width 3 px.

### Map mode

현재 `"rgb"` 단일 모드 사용. `"hybrid"` 모드(채널별 semantic 분리)는 구현은 되어 있으나 **현 연구에서 사용하지 않기로 결정** (Research Note 2026-05-18 참조). 모델이 이미지 자체를 이해한다고 보고 rgb로 충분하다고 판단. 후속 연구에서 재검토.

### Route generation

두 가지 라우팅 방식이 공존:

| 컨텍스트 | 방식 | 이유 |
|----------|------|------|
| **FrodoBots 데이터셋 빌드** | `osm_snap.py` — Overpass API + Dijkstra | 서버 불필요, 오프라인 캐시, 보행자 전용 |
| **rosbag 학습** | OSRM `/match` | EKF GPS, 실내외 혼합 경로 |
| **추론 (ROS)** | OSRM `/route` | 미션 시작 시 1회 호출 후 캐시 |

**osm_snap.py (FrodoBots 전용)**:
1. Overpass API → 보행자 네트워크 다운로드 (footway/path/pedestrian/steps 등), 결과 JSON 캐시
2. `snap_trajectory()` → KDTree 최근접 엣지 투영, 평균 거리 반환 (품질 필터)
3. `snap_trajectory_graph()` → 같은 엣지 연속: 직접 투영 / 엣지 전환: Dijkstra 코너 노드 경유 (90° 꺾임의 빗변 아티팩트 제거)

### `configs/map.yaml` (shared across all pipelines)
```yaml
tile_url: "https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
tile_zoom: 19
render_size: 512       # canvas before rotation
output_size: 224       # final image side length
mode: "rgb"            # "hybrid" 구현됨, 현재 미사용
crop_ratio: 0.7        # 중앙 70% 크롭 후 업스케일 → 1.43× 줌 효과
osrm_url: "https://router.project-osrm.org"
osrm_profile: "foot"
osrm_valid_threshold_m: 10.0
```

### `configs/paths.yaml` (environment-specific)
```yaml
frodo_root:   /home/hmmdyn/data/frodobots   # FrodoBots 원본 데이터 루트
dataset_root: /home/hmmdyn/data/frodobots/dataset   # 출력 데이터셋 루트
tile_cache:   ...dataset/osm_cache          # 타일 캐시
osm_cache:    ...dataset/osm_net_cache      # Overpass 결과 캐시
```

env var으로 오버라이드 가능: `DYNAV_FRODO_ROOT`, `DYNAV_DATASET_ROOT`, `DYNAV_TILE_CACHE`, `DYNAV_OSM_CACHE`

---

## Loss Functions

```
L_total = L_waypoint + λ1·L_direction + λ2·L_progress + λ3·L_smooth

L_waypoint  = (1/H) Σ ||â_i - a*_i||_p           # p=1(L1/MAE) or p=2(L2/MSE), config 선택
L_direction = 1 - cos(α̂, α*_route)               # route alignment
L_progress  = -(1/H) Σ (â_i · d̂_route)           # progress incentive
L_smooth    = (1/(H-1)) Σ ||â_{i+1} - â_i||²     # smoothness
```

waypoint loss 종류(`loss.waypoint_type`: `"l1"`/`"l2"`), λ1·λ2·λ3, enable 플래그 값은 모두 **`configs/default.yaml`의 `loss.*`가 권위** (`waypoint_type`, `lambda_direction`/`lambda_progress`/`lambda_smooth`, `enable_direction`/`progress`/`smooth`). 이 문서는 수식 형태만 고정하고 수치·선택은 config를 따른다.

## Training Loop (`scripts/train.py`)

- **Optimizer/scheduler:** AdamW + linear warm-up → cosine decay. `lr`·`weight_decay`·`warmup_epochs`는 `configs/default.yaml`의 `training.*`가 권위.
- **Encoder freeze:** 첫 `encoder.freeze_epochs` 동안 backbone freeze 후 unfreeze.
- **Validation:** `_eval_one_epoch`이 component별 평균 loss dict 반환 → WandB에 `val/{waypoint,direction,progress,smooth,total}` 기록 (train과 대칭).
- **Early stopping:** val total loss가 `training.early_stopping_patience` epoch 동안 개선되지 않으면 중단 (`0`이면 비활성). best는 `checkpoints/best.pt`.
- **AMP:** `training.amp` + CUDA일 때 BF16 autocast.

---

## File Structure

```
dynav/
├── CLAUDE.md
├── configs/
│   ├── map.yaml                          # shared map config (tile, render, crop_ratio)
│   ├── paths.yaml                        # environment-specific data paths
│   ├── default.yaml                      # model hyperparameters
│   ├── rosbag_topics.yaml
│   ├── record_topics.yaml
│   ├── ablation_map_hybrid.yaml          # hybrid mode (구현됨, 현재 미사용)
│   ├── ablation_no_direction.yaml
│   ├── ablation_no_progress.yaml
│   └── ablation_waypoint_only.yaml
├── dynav/
│   ├── map/                              # unified map module
│   │   ├── __init__.py
│   │   ├── tiles.py                      # TileCache, stitch_tiles
│   │   ├── routing.py                    # OSRMRouter (추론용 유지)
│   │   ├── osm_snap.py                   # Overpass+Dijkstra 오프라인 스냅
│   │   ├── segment.py                    # segment_gps_episode
│   │   ├── overlay.py                    # visual constants + drawing
│   │   └── renderer.py                   # MapRenderer (crop_ratio 파라미터 추가됨)
│   ├── models/
│   │   ├── encoders/ (visual_encoder.py, map_encoder.py)
│   │   ├── decoders/ (cross_attention_decoder.py, self_attention_decoder.py, action_heads.py)
│   │   └── map_nav_model.py
│   ├── data/ (dataset.py, transforms.py)
│   ├── losses/ (navigation_losses.py)
│   └── utils/ (geometry.py, visualization.py)
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize_attention.py
│   ├── record_bag.py
│   ├── extract_rosbag.py                 # rosbag → training samples (OSRM)
│   ├── extract_frodobots_segments.py     # Step 1: raw rides → valid_segments_{group}.json
│   ├── extract_frodobots_frames.py       # Step 2: valid_segments → stride-10 JPEGs
│   ├── build_frodobots_dataset.py        # Step 3: segments+frames → samples+GIFs
│   ├── dynav_gui.py                      # 파이프라인 GUI (3단계 통합)
│   └── view_samples.py                   # 샘플 빠른 뷰어
└── tests/
    ├── test_map.py
    ├── test_encoders.py
    ├── test_decoders.py
    ├── test_model.py
    └── test_losses.py
```

---

## Coding Conventions

- Type hints on all function signatures.
- Tensor shape comments: `# (B, N, d)` inline.
- Config-driven via Hydra/OmegaConf. All hyperparameters in YAML.
- All modules independently testable.
- `einops.rearrange` for complex tensor ops.

---

## Key Implementation Notes

### EfficientNet-B0
```python
# model.features → (B, 1280, 7, 7)
# VisualEncoder: features → GAP → (B, 1280) → project → (B, d)     [per obs image]
# MapEncoder:   features → GAP → (B, 1280, 1, 1) → project → (B, 1, d)
```

### Heading-up rotation
```python
# PIL.rotate(heading_deg, expand=False) — PIL CCW-positive
# Compass heading (CW) passed directly → rotates canvas so heading points up
canvas.rotate(heading_deg, center=robot_px, expand=False)
```

### route_direction (body frame, radians)
```python
enu_heading = math.radians(90.0 - heading_deg)   # compass CW → ENU CCW
route_dir = compute_route_direction(
    np.array([lat, lon]), np.array(matched_route),
    enu_heading, lookahead_distance=10.0,
)  # returns radians in body frame
```

### FrodoBots-7k 데이터셋 빌드 파이프라인 (신규, `dynav/frodo7k/`)

LeRobot v1.6 포맷(`dataset_cache.zarr` 164.7M frames @10fps, 187,156 episodes,
39,856 rides, 224×128 front/rear video GOP=2)을 직접 소비한다. 구버전 3단계
파이프라인을 대체한다.

```
python scripts/build_frodobots7k_dataset.py --stage index [--limit N | --episodes a:b]
python scripts/build_frodobots7k_dataset.py --stage select
python scripts/build_frodobots7k_dataset.py --stage build
python scripts/build_frodobots7k_dataset.py --stage report
```

모든 임계값은 `configs/frodo7k.yaml`이 권위. 출력: `<output_root>/{index,train,val,report}`.

**GT 기하 (구버전 waypoint 불일치의 해결):**
- `observation.filtered_position` (EKF, 로컬 ENU m) + `observation.filtered_heading`
  (rad, East 기준 CCW)에서 직접 산출. raw GPS 보간·±2s bearing 근사 미사용.
- body frame: x=fwd=(cos h, sin h), y=left=(−sin h, cos h).
  compass_deg = (90 − deg(h)) % 360.
- waypoint: 1–5 s (frames [10..50] @10fps), meta에 정규화값(`waypoint_norm_m: 5.0`)과
  raw 미터(`gt_waypoints_m`) 모두 저장. (기존 2.5 m 정규화는 5 s waypoint를 상시 클리핑했음.)

**stage index** — 에피소드 → `refine_segments` (video ts 갭 >0.25 s·5 s 정지·EKF↔GPS
발산으로 분할, 정지 head/tail 트림) → `segment_qa` 게이트 (arc·net_disp·speed·
stationary·gps_dev_p95·**heading↔motion 일치**; straightness 게이트 없음 — 회전 데이터 보존)
→ OSM 라우팅: `fetch_network_bbox` (`_DRIVE_PED_TAGS` 확장 태그 + 세그먼트 전체 bbox,
0.005° 그리드 캐시) → **`route_by_graph` 그래프 길찾기** — 시작·골만 방향 인지 투영
(데모 접선과 >50° 어긋난 엣지 배제 → 교차로 측면도로 오스냅 방지) 후 Dijkstra 1회,
데모가 최단경로에서 >10 m 벗어난 지점에만 via-앵커(그래프 **노드** 스냅)를 재귀 삽입.
근접 노드 2.5 m 스티칭(보도·차도 그래프 분절 해소) + 스퍼 제거 후처리.
inference `/route`와 동일한 "도로 중심선 그래프 경로" 스타일 = train/inference 분포 일치.
→ `route_qa` (snap dist·len ratio·**offroute gap**(데모에서 먼 장거리 점프만)·monotonic·
tangent 방향 일치 ≤45°·**spike=0**) → scene 분류 (`fetch_scene_bbox`: 녹지 폴리곤 PIP
+ 건물 수 `out count` → park/city/straight_road/other) → `candidate_indices`
(직진 stride 20 / 회전 stride 5) → `classify_candidates` (maneuver 6종·
near_intersection·env_density·scene·difficulty). 에피소드당 JSON 1개, resumable.

**stage select** — ride md5 해시 train/val 분할(샘플 누수 방지) → `select_balanced`:
**(maneuver × scene) 2축 쿼터** (`maneuver_targets` × `scene_targets`) + 세그먼트당 캡.
부족 셀은 같은 maneuver 행 내 재분배 → 전역 재분배. `selection_stats.json`에 기록.

**stage build** — PyAV로 에피소드당 1회 디코드 → obs_0..3 (0.5 s stride) +
`MapRenderer` (heading-up, 경로는 로봇의 **폴리라인 투영점**부터 슬라이스 — 그래프
경로는 정점이 희소하므로 정점 거리가 아닌 `_project_to_polyline` 세그먼트 투영 필수).
샘플 게이트: `wp_backward`, `robot_far_from_route`(폴리라인 거리 >15 m),
`route_wp_conflict`(GT 진행방향↔경로방향 >75° — OSM에 없는 골목 주행 등 지도·GT 모순 샘플 제거).
meta에 `route_lateral_m`(로봇↔경로 오프셋) 기록.
출력 포맷은 DyNavDataset과 동일(`obs_0..3.png, map.png, meta.json`) + meta에 라벨·QA 전부 포함.

**inference 계약 (이탈 대응)** — policy는 lateral offset ≤15 m 범위에서만 동작을
보장한다(학습 게이트와 동일). ROS 노드는 경로 이탈 ~12 m가 지속되면 현재위치→골을
재라우팅한다. 소·중간 오프셋 복귀는 자연 데이터(lateral med ~3 m, p90 ~7 m) +
학습 시 map augmentation(RandomAffine 이동·회전 = GPS/heading 오차 시뮬레이션)이 커버.

**stage report** — `report.md` + contact sheet (obs|map+waypoint 오버레이) + 구성 통계.

### (구) FrodoBots 데이터셋 빌드 파이프라인 (3단계, deprecated — rides0~4 CSV 입력 전용)

**Step 1 — 세그먼트 추출** (`extract_frodobots_segments.py`):
```
output_rides_*/{gps,camera_timestamps}.csv
→ stride-10 프레임 GPS 보간 + smoothed speed
→ 정지 감지 (speed < 0.4 m/s, 3초 지속) / GPS 갭 (>5s) 으로 분할
→ valid_segments_{group}.json
   { ride_id: { segments: [{seg_idx, frame_ids, frame_lat/lon, n_frames,
                             net_disp_m, avg_speed_ms, straightness}] } }
```
파라미터 env var: `DYNAV_SEG_STOP_SPEED` `DYNAV_SEG_STOP_WINDOW` `DYNAV_SEG_GPS_GAP` `DYNAV_SEG_MIN_FRAMES`  
incremental: 이미 처리된 ride는 skip.

**Step 2 — 프레임 추출** (`extract_frodobots_frames.py`):
```
valid_segments_{group}.json + output_rides_*/recordings/*.m3u8
→ ffmpeg로 stride-10 프레임 추출
→ dataset/frames/ride_{id}/{frame_id:06d}.jpg
```
segment frame_ids에 필요한 프레임 + obs lookback(×3)만 보존, 나머지 삭제.  
이미 필요한 프레임이 모두 있으면 skip. `DYNAV_FRAME_QUALITY` `DYNAV_FRAME_FORCE`

**Step 3 — 데이터셋 빌드** (`build_frodobots_dataset.py`):
```
frodo_root/valid_segments_*.json (glob, 자동 탐색) + dataset/frames/
→ quality pre-filter (net_disp, speed, straightness)
→ fetch_ped_network() (Overpass, 캐시) + snap_trajectory() (OSM snap < 10m)
→ snap_trajectory_graph() (Dijkstra 코너 보정)
→ MapRenderer.render() → map.png
→ obs_0..3.png (stride-10 × 3 lookback)
→ GIF + manifest.json
```
필터 파라미터 env var: `DYNAV_OSM_SNAP_THRESH` `DYNAV_NET_DISP` `DYNAV_SPEED` `DYNAV_STRAIGHTNESS` `DYNAV_SAMPLE_STRIDE`

**현재 생성된 데이터셋 (2026-05-18):**  
rides0~4 기반 후처리 스크립트 수동 구동으로 필터링 → train 4,483개 + val 502개 = 4,985개 샘플 (이 데이터로 학습 진행)

---

## Ablation Configs

| File | Effect |
|------|--------|
| `ablation_map_hybrid.yaml` | `map.mode: "hybrid"` — 현재 미사용, 후속 연구용으로 보류 |
| `ablation_no_direction.yaml` | `enable_direction: false` |
| `ablation_no_progress.yaml` | `enable_progress: false` |
| `ablation_waypoint_only.yaml` | all auxiliary losses off |

Usage: `python scripts/train.py --config-name ablation_no_direction`

---

## Validation

```bash
# Unit tests
pytest tests/test_map.py -v

# Visual sanity check
python -c "
from dynav.map import MapRenderer, TileCache, OSRMRouter
from dynav.map.routing import is_route_valid
cache  = TileCache('/tmp/dynav_tile_cache')
router = OSRMRouter()
r_rgb    = MapRenderer(cache, zoom=19, output_size=224, mode='rgb')
r_hybrid = MapRenderer(cache, zoom=19, output_size=224, mode='hybrid')
gps = [(37.2880 + i*0.0001, 126.9759) for i in range(20)]
route, dev = router.match(gps)
print(f'valid={is_route_valid(route, gps)} dev={dev:.1f}m')
kw = dict(lat=37.2890, lon=126.9759, heading_deg=45.0,
          route_latlons=route, goal_lat=37.2920, goal_lon=126.9780)
r_rgb.render(**kw).save('/tmp/test_rgb.png')
r_hybrid.render(**kw).save('/tmp/test_hybrid.png')
print('saved /tmp/test_rgb.png /tmp/test_hybrid.png')
"
```
