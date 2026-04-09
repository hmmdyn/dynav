# CLAUDE.md — Map Navigation Model (dynav)

## Project Overview

This project implements a lightweight dual-mode navigation model for outdoor robot navigation on edge devices (NVIDIA Jetson Orin Nano Super, 8GB RAM, 67 TOPS). The robot platform is a Clearpath Jackal UGV (4×4 differential drive, cmd_vel interface).

The system consists of two models:
1. **Map Navigation Model** (this phase): Uses a rendered OpenStreetMap image with route overlay + egocentric camera observations to predict relative waypoints for long-range navigation (500m–1km).
2. **Semantic Navigation Model** (future phase): Uses CLIP+FiLM language conditioning + egocentric observations for last-mile navigation to a specific target.

**Current Phase: Phase 1 — Data collection pipeline (Insta360 + Jackal J100) complete; ready for real-world training.**

## Architecture Summary

### Inputs
- **Observations:** 4 images total — front camera (current + 2 past frames) + rear camera (current). Each 224×224 RGB.
- **Map+Path Image:** 1 image — OSM-rendered top-down map centered on robot, with route drawn in red, robot position as blue arrow, destination as green marker. 224×224 RGB.

### Encoders (separate weights, both EfficientNet-B0 based)
- **VisualEncoder:** Each observation image → EfficientNet-B0 → Global Average Pooling → Linear(1280, 256) → 1 token. 4 images → `obs_tokens ∈ (B, 4, 256)`.
- **MapEncoder:** Map image → EfficientNet-B0 → AdaptiveAvgPool2d((3,3)) → Linear(1280, 256) → 9 tokens + 2D positional encoding → `map_tokens ∈ (B, 9, 256)`.
  - Positional encoding type is configurable: `"learnable"` (default, `nn.Parameter`) or `"sinusoidal"` (fixed buffer, `_build_2d_sinusoidal_encoding`). Set via `encoder.map_pos_enc`.

### Decoder (two options, selectable via config)
- **CrossAttentionDecoder:** [Self-Attention on obs] → [Cross-Attention: Q=obs, K/V=map] → [FFN]. Repeated L=4 times. Output: `context ∈ (B, 256)` via mean pooling.
- **SelfAttentionDecoder:** Concatenate obs+map tokens → [Self-Attention on all] → [FFN]. Repeated L=4 times. Output: `context ∈ (B, 256)` via mean pooling over obs token positions. This is the ViNT-style baseline for ablation comparison.

### Action Head
- **WaypointHead:** Linear(256, 128) → ReLU → Linear(128, H×2) → reshape (H, 2). Predicts H=5 relative waypoints (Δx, Δy) in robot body frame.

### Output
- `waypoints ∈ (B, H, 2)` — H relative (Δx, Δy) pairs in robot body frame, normalized to [-1, 1].

## Loss Functions (in `dynav/losses/navigation_losses.py`)

```
L_total = L_waypoint + λ1 * L_direction + λ2 * L_progress + λ3 * L_smooth

L_waypoint = (1/H) * Σ ||â_i - a*_i||₁           # L1 waypoint regression
L_direction = 1 - cos(α̂, α*_route)                # route direction alignment
L_progress  = -(1/H) * Σ (â_i · d̂_route)          # route progress incentive
L_smooth    = (1/(H-1)) * Σ ||â_{i+1} - â_i||²    # trajectory smoothness

Default: λ1=0.5, λ2=0.1, λ3=0.01
```

Each auxiliary term can be individually disabled via config flags (`enable_direction`, `enable_progress`, `enable_smooth`). When disabled the term is 0.0 and excluded from the total. See ablation configs below.

## Key Design Decisions

1. **Map encoder uses spatial tokens (9), visual encoder uses GAP tokens (4).** Map's spatial layout (where the route goes) is critical information that GAP would destroy. 3×3 adaptive pooling (9 tokens) captures coarse direction (left/center/right) while keeping the obs:map token ratio reasonable (4:9). Observations need high-level semantics, so GAP suffices.
2. **Q=obs, K/V=map in cross-attention.** "Observations query the map for guidance." The robot's current visual situation asks the map where to go.
3. **No temporal distance prediction** (unlike ViNT). ViNT's temporal distance is for topological graph reachability, which this model doesn't use. Replaced by direction and progress losses.
4. **Decoder files are separate** so that cross-attention and self-attention approaches can be compared via config switch (`decoder_type: "cross_attention" | "self_attention"`).

## Tech Stack

- Python 3.10+
- PyTorch 2.1+
- torchvision (EfficientNet-B0 pretrained weights)
- einops (tensor reshaping)
- hydra-core + OmegaConf (config management)
- wandb (experiment logging)

## File Structure

```
dynav/
├── CLAUDE.md                     # THIS FILE
├── configs/
│   ├── default.yaml              # shared hyperparameters
│   ├── ablation_no_direction.yaml
│   ├── ablation_no_progress.yaml
│   ├── ablation_waypoint_only.yaml
│   ├── rosbag_topics.yaml        # extract_rosbag.py config (Insta360 + J100 defaults)
│   └── record_topics.yaml        # record_bag.py config (topics to record)
├── dynav/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   ├── visual_encoder.py
│   │   │   └── map_encoder.py
│   │   ├── decoders/
│   │   │   ├── __init__.py
│   │   │   ├── cross_attention_decoder.py
│   │   │   ├── self_attention_decoder.py
│   │   │   └── action_heads.py
│   │   └── map_nav_model.py      # assembles encoder+decoder+head
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # DyNavDataset, DummyDyNavDataset
│   │   └── transforms.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── navigation_losses.py
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py
│       └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── sanity_check.py
│   ├── visualize_attention.py    # --per-head flag for per-head heatmaps
│   ├── record_bag.py             # config-driven rosbag recording (Insta360 + J100)
│   └── extract_rosbag.py         # rosbag → training samples
└── tests/
    ├── test_encoders.py
    ├── test_decoders.py
    ├── test_model.py
    └── test_losses.py
```

## Coding Conventions

- Type hints on all function signatures.
- Docstrings (Google style) on all public classes and methods.
- Tensor shape comments: `# (B, N, d)` format inline.
- Config-driven: all hyperparameters in YAML, accessed via OmegaConf.
- No code copied from ViNT/NoMaD/LeLaN repos. Architecture is inspired by those papers but implemented independently.
- Use `einops.rearrange` for complex tensor reshaping instead of manual view/permute chains.
- All modules should work independently — encoders, decoders, heads can be tested in isolation.

## Implementation Notes

### EfficientNet-B0 Feature Extraction
```python
# torchvision EfficientNet-B0 structure:
# model.features → Sequential of blocks, output: (B, 1280, 7, 7)
# model.avgpool → AdaptiveAvgPool2d, output: (B, 1280, 1, 1)
# model.classifier → Linear(1280, 1000)
#
# For VisualEncoder: use features + avgpool → flatten → (B, 1280) → project to (B, d)
# For MapEncoder: use features only → (B, 1280, 7, 7) → AdaptiveAvgPool2d((3,3)) → (B, 1280, 3, 3) → reshape to (B, 9, 1280) → project to (B, 9, d)
```

### 2D Positional Encoding for Map Tokens
Map spatial tokens need positional encoding to preserve grid layout. Two options:
```python
# Learnable (default): nn.Parameter(torch.randn(1, 9, d) * 0.02)  — trainable
# Sinusoidal (fixed):  _build_2d_sinusoidal_encoding(grid_size=3, dim=d)
#   → splits dim into two halves: first half = row encoding, second = col encoding
#   → each half = sin+cos at quarter_dim frequency bands
#   → registered as buffer (not trained)
# Select via: encoder.map_pos_enc: "learnable" | "sinusoidal"
```

### Cross-Attention Decoder Block Structure
```
Input: obs_tokens (B, N_o, d), map_tokens (B, N_m, d)
│
├─ Sub-layer 1: MultiHeadSelfAttention(obs_tokens) + LayerNorm + residual
│  → obs_tokens' (B, N_o, d)
│
├─ Sub-layer 2: MultiHeadCrossAttention(Q=obs_tokens', K=map_tokens, V=map_tokens) + LayerNorm + residual
│  → obs_tokens'' (B, N_o, d)
│  → attn_weights:
│      averaged (return_attention=True):  (B, N_o, N_m)
│      per-head  (return_per_head=True):  (B, n_heads, N_o, N_m)
│
├─ Sub-layer 3: FFN(obs_tokens'') + LayerNorm + residual
│  → obs_tokens_out (B, N_o, d)
│
Output: obs_tokens_out, attn_weights
```

Per-head weights are available throughout: `CrossAttentionDecoderBlock` → `CrossAttentionDecoder` → `DyNavModel.forward(return_per_head=True)`. Visualized with `scripts/visualize_attention.py --per-head`.

### Self-Attention Decoder Block Structure (ViNT-style baseline)
```
Input: obs_tokens (B, N_o, d), map_tokens (B, N_m, d)
│
├─ Concatenate: tokens = cat([obs_tokens, map_tokens], dim=1) → (B, N_o+N_m, d)
│
├─ Sub-layer 1: MultiHeadSelfAttention(tokens) + LayerNorm + residual
│  → tokens' (B, N_o+N_m, d)
│
├─ Sub-layer 2: FFN(tokens') + LayerNorm + residual
│  → tokens_out (B, N_o+N_m, d)
│
├─ Extract obs positions: tokens_out[:, :N_o, :] → (B, N_o, d)
│
Output: obs_tokens_out (B, N_o, d)
```

## Config Schema (configs/default.yaml)

```yaml
model:
  token_dim: 256                    # d — embedding dimension
  obs_context_length: 2             # K — number of past frames
  prediction_horizon: 5             # H — number of waypoints to predict

encoder:
  backbone: "efficientnet_b0"
  pretrained: true
  freeze_epochs: 5                  # freeze encoder for first N epochs
  map_pos_enc: "learnable"          # "learnable" or "sinusoidal"

decoder:
  type: "cross_attention"           # "cross_attention" or "self_attention"
  n_layers: 4
  n_heads: 4
  d_ff: 512
  dropout: 0.1

action_head:
  type: "regression"                # "regression" or "diffusion" (future)
  hidden_dim: 128

loss:
  enable_direction: true            # set false to ablate direction loss
  enable_progress: true             # set false to ablate progress loss
  enable_smooth: true               # set false to ablate smooth loss
  lambda_direction: 0.5
  lambda_progress: 0.1
  lambda_smooth: 0.01

training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-4
  optimizer: "adamw"
  epochs: 100
  scheduler: "cosine"
  warmup_epochs: 5

data:
  image_size: 224
  map_image_size: 224
  normalize_waypoints: true
  max_waypoint_distance: 2.5        # meters, for normalization
```

## Ablation Configs

Three ready-made ablation overrides in `configs/`:

| File | Effect |
|---|---|
| `ablation_no_direction.yaml` | `enable_direction: false` |
| `ablation_no_progress.yaml` | `enable_progress: false` |
| `ablation_waypoint_only.yaml` | All auxiliary losses disabled |

Usage: `python scripts/train.py --config-name ablation_waypoint_only`
