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
# hybrid mode Gaussian parameters:
ROUTE_SIGMA_PX     = 8.0
ROBOT_SIGMA_PX     = 8.0
GOAL_SIGMA_PX      = 5.0
GOAL_AMPLITUDE     = 0.5
```

Past route (already-traversed portion) is **not shown** — route displayed from current position to goal only.

### Marker spec
- **Robot**: blue filled circle, radius 6 px. Heading-up is fixed → direction arrow is redundant.
- **Goal**: green filled circle, radius 5 px.
- **Route**: red polyline, current_idx → end, width 3 px (rgb). Gaussian soft mask in hybrid Ch G.

### Map modes
| Mode | Ch R | Ch G | Ch B |
|------|------|------|------|
| `"rgb"` | OSM tile (R) | OSM tile (G) | OSM tile (B) |
| `"hybrid"` | OSM grayscale | Gaussian route mask [0,1]→[0,255] | Robot(amp=1.0) + Goal(amp=0.5) Gaussian [0,1]→[0,255] |

### Route generation (train/inference consistency)
- **Training (rosbag)**: GPS trajectory → OSRM `/match` → OSM-snapped route. Called once per episode.
- **Training (FrodoBots)**: segment GPS → OSRM `/match` (radius=50m for 1Hz GPS). Called once per segment.
- **Inference**: current + goal → OSRM `/route`. Called once at mission start, cached.

Both use OSM network routes → train/inference distribution match.

### `configs/map.yaml` (shared across all pipelines)
```yaml
tile_url: "https://cartodb-basemaps-a.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png"
tile_zoom: 19
render_size: 512       # canvas before crop/resize
output_size: 224       # final image side length
mode: "rgb"            # "rgb" | "hybrid"
osrm_url: "https://router.project-osrm.org"
osrm_profile: "foot"
osrm_match_radius_m: 25.0           # rosbag (EKF GPS)
osrm_match_radius_frodobots: 50.0   # FrodoBots (1Hz noisy GPS)
osrm_valid_threshold_m: 10.0
route_sigma_px: 8.0   # hybrid mode only
```

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
│   ├── map.yaml                          # shared map/OSRM config
│   ├── default.yaml                      # model hyperparameters (includes map)
│   ├── rosbag_topics.yaml                # extract_rosbag.py topic config
│   ├── record_topics.yaml                # record_bag.py topic config
│   ├── ablation_map_hybrid.yaml          # hybrid map mode ablation
│   ├── ablation_no_direction.yaml
│   ├── ablation_no_progress.yaml
│   └── ablation_waypoint_only.yaml
├── dynav/
│   ├── map/                              # unified map module (NEW)
│   │   ├── __init__.py
│   │   ├── tiles.py                      # TileCache, stitch_tiles
│   │   ├── routing.py                    # OSRMRouter, is_route_valid, find_current_idx
│   │   ├── segment.py                    # segment_gps_episode (FrodoBots)
│   │   ├── overlay.py                    # visual constants + rgb/hybrid drawing
│   │   └── renderer.py                   # MapRenderer (assembles everything)
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
│   ├── extract_rosbag.py                 # rosbag → training samples
│   └── build_frodobots_dataset.py        # FrodoBots v1+v2 unified pipeline
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

### FrodoBots segment filtering
`segment_gps_episode()` pipeline: GPS jump split → stationary removal → loop detection → min-length filter (10m).

---

## Ablation Configs

| File | Effect |
|------|--------|
| `ablation_map_hybrid.yaml` | `map.mode: "hybrid"`, per-channel normalize |
| `ablation_no_direction.yaml` | `enable_direction: false` |
| `ablation_no_progress.yaml` | `enable_progress: false` |
| `ablation_waypoint_only.yaml` | all auxiliary losses off |

Usage: `python scripts/train.py --config-name ablation_map_hybrid`

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
