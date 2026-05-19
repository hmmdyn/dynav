# CLAUDE.md — Map Navigation Model (dynav)

## Project Overview

Lightweight dual-mode navigation model for outdoor robot navigation on edge devices (NVIDIA Jetson Orin Nano Super, 8GB RAM, 67 TOPS). Robot platform: Clearpath Jackal UGV (4×4 differential drive, cmd_vel interface).

**Phase 1 — Data pipeline complete; ready for real-world training.**

Two planned models:
1. **Map Navigation Model** (this phase): OSM map image + route overlay + egocentric camera → relative waypoints for long-range navigation (500m–1km).
2. **Semantic Navigation Model** (future): CLIP+FiLM language conditioning + egocentric observations for last-mile navigation.

---

## Architecture

### Inputs
- **Observations:** 4 images — front camera (current + 2 past) + rear camera (current). Each 224×224 RGB.
- **Map+Path Image:** 224×224 RGB — OSM top-down map, heading-up, route in red, robot as blue circle, goal as green circle.

### Encoders
- **VisualEncoder:** Each obs image → EfficientNet-B0 → GAP → Linear(1280, 256) → 1 token. 4 images → `obs_tokens ∈ (B, 4, 256)`.
- **MapEncoder:** Map image → EfficientNet-B0 → AdaptiveAvgPool2d(3×3) → Linear(1280, 256) → 9 tokens + 2D pos enc → `map_tokens ∈ (B, 9, 256)`.
  - Pos enc: `"learnable"` (default, `nn.Parameter`) or `"sinusoidal"` (fixed buffer).

### Decoder
- **CrossAttentionDecoder** (default): Self-Attn(obs) → Cross-Attn(Q=obs, K/V=map) → FFN × 4. Output: `context ∈ (B, 256)` via mean pool.
- **SelfAttentionDecoder** (ViNT-style baseline): Cat(obs, map) → Self-Attn → FFN × 4. Output: mean pool over obs positions.

### Action Head
`WaypointHead`: Linear(256, 128) → ReLU → Linear(128, H×2) → reshape `(H, 2)`.  
Output: H=5 relative waypoints (Δx, Δy) in robot body frame, normalized to [-1, 1].

---

## Map Module (`dynav/map/`)

Self-contained package. No external submodules (osmnav removed).

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

L_waypoint  = (1/H) Σ ||â_i - a*_i||₁           # L1 regression
L_direction = 1 - cos(α̂, α*_route)               # route alignment
L_progress  = -(1/H) Σ (â_i · d̂_route)           # progress incentive
L_smooth    = (1/(H-1)) Σ ||â_{i+1} - â_i||²     # smoothness

λ1=0.5, λ2=0.1, λ3=0.01 (defaults)
```

Each term individually disabled via `enable_direction/progress/smooth` config flags.

---

## File Structure

```
dynav/
├── CLAUDE.md
├── configs/
│   ├── map.yaml                          # shared map config (tile, render, crop_ratio)
│   ├── paths.yaml                        # environment-specific data paths (NEW)
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
│   │   ├── osm_snap.py                   # Overpass+Dijkstra 오프라인 스냅 (NEW)
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
│   ├── sanity_check.py
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
# VisualEncoder: features + avgpool → (B, 1280) → project → (B, d)
# MapEncoder:   features → AdaptiveAvgPool2d(3,3) → (B, 9, 1280) → project → (B, 9, d)
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

### FrodoBots 데이터셋 빌드 파이프라인 (3단계)

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
valid_segments_*.json + dataset/frames/
→ quality pre-filter (net_disp, speed, straightness)
→ fetch_ped_network() (Overpass, 캐시) + snap_trajectory() (OSM snap < 10m)
→ snap_trajectory_graph() (Dijkstra 코너 보정)
→ MapRenderer.render() → map.png
→ obs_0..3.png (stride-10 × 3 lookback)
→ GIF + manifest.json
```
필터 파라미터 env var: `DYNAV_OSM_SNAP_THRESH` `DYNAV_NET_DISP` `DYNAV_SPEED` `DYNAV_STRAIGHTNESS` `DYNAV_SAMPLE_STRIDE`

**현재 생성된 데이터셋 (2026-05-18):**  
train 28,081개 + val 3,421개 = 31,502개 샘플 (rides0, rides2 기반)  
rides22, rides23 신규 데이터 추가 예정.

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
