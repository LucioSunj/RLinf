# DINOv3 Visual Similarity Reward Model

This document describes the DINOv3-based visual similarity reward model integrated into RLinf for embodied robot manipulation tasks.

## 1. Core Idea

The reward model computes a scalar reward by comparing the **visual similarity** between the current environment observation and an **expert reference image** showing the desired final state.

Key insight: DINOv3 (a self-supervised vision foundation model) produces semantically meaningful embeddings that capture high-level scene structure. Two images depicting the same task outcome will have similar embeddings, even if pixel-level details differ (lighting, viewpoint, texture).

Use cases:
- Provide dense/terminal rewards for RL training when only a single expert demonstration image is available.
- Replace hand-crafted reward functions for manipulation tasks.
- Enable reward shaping based on visual progress toward a goal state.

## 2. Mathematical Formulation

### 2.1 Feature Extraction

Given an image \(x\), DINOv3 produces a feature vector via the CLS token:

```
f(x) = normalize(DINOv3(x).last_hidden_state[:, 0])  # L2-normalized, unit hypersphere
```

The embedding lies on the unit hypersphere: \(\|f(x)\|_2 = 1\).

### 2.2 Similarity Metric

For current observation \(x_{\text{cur}}\) and expert reference \(x_{\text{exp}}\):

```
d^2 = ||f(x_cur) - f(x_exp)||^2          # squared L2 distance, range [0, 4]
```

Since both vectors are L2-normalized, the maximum possible distance is 4 (antipodal points), and the minimum is 0 (identical embeddings).

### 2.3 RBF Kernel Reward

The reward is computed via an RBF (Gaussian) kernel:

```
R = exp(-d^2 / T)
```

Where:
- \(T\): temperature hyperparameter controlling sensitivity.
- Default \(T = 1.0\) gives reward range \([e^{-4}, 1] \approx [0.018, 1]\).
- Higher \(T\): reward is less sensitive to distance (smoother).
- Lower \(T\): reward drops off faster (more discriminative).

### 2.4 Why RBF?

- Bounded output: \(R \in (0, 1]\), naturally interpretable as a probability-like score.
- Monotonic: closer embeddings give higher rewards.
- Differentiable: compatible with end-to-end training if needed.
- No learned parameters in the reward computation itself: the DINOv3 backbone is frozen.

## 3. Code Distribution

### 3.1 Model Implementation

**File**: `rlinf/models/embodiment/reward/dinov3_reward_model.py`

Implements `DINoV3RewardModel(BaseImageRewardModel)`.

Key methods:
- `__init__(cfg)`: Loads DINOv3 backbone and image processor from HuggingFace (or local path). Precomputes and caches the expert reference embedding as a registered buffer.
- `_get_embedding(pil_images)`: Extracts L2-normalized CLS token embeddings in eval mode (no grad).
- `forward(input_data, labels)`: Accepts NHWC uint8 or NCHW float tensors, converts to PIL, extracts embeddings, computes RBF reward. Returns dict with `loss`, `accuracy`, `logits`, `probabilities` (all required by RLinf interface).
- `compute_reward(obs_dict)`: Wrapper that accepts `{"images": ...}` dict and returns reward tensor.

Design choices:
- **Frozen backbone**: `requires_grad=False` on all DINOv3 parameters. The reward model is not fine-tuned.
- **Precomputed expert embedding**: Computed once at init, cached as a buffer to avoid redundant forward passes.
- **Batch processing**: Supports batched image inputs for efficiency.
- **Dual input format**: Handles both NHWC uint8 (env worker format) and NCHW float (post-processing format).

### 3.2 Model Registration

**File**: `rlinf/models/embodiment/reward/__init__.py`

The model is registered in `reward_model_registry`:

```python
reward_model_registry = {
    "resnet": ResNetRewardModel,
    "dinov3": DINoV3RewardModel,
}
```

This enables config-driven instantiation via `model_type: "dinov3"`.

### 3.3 Reward Worker Integration

**File**: `rlinf/workers/reward/reward_worker.py`

The reward worker receives observations from env workers and forwards them to the reward model.

Key integration point (line ~777 in `env_worker.py`):

```python
reward_input = {"images": reward_input_obs["main_images"]}
```

Env workers automatically extract `main_images` from Robotwin/ManiSkill observations and pass them as `"images"` to the reward worker.

The reward worker supports two modes:
- `continuous_reward=True`: Returns raw similarity scores (probabilities) from the model.
- `continuous_reward=False`: Binarizes rewards via threshold (`reward_threshold`).

For DINOv3 similarity rewards, `continuous_reward=True` is recommended to preserve the full gradient signal.

### 3.4 Standalone Calculator (Optional)

**File**: `examples/reward/dino_reward_calculator.py`

A standalone utility for local testing outside of RLinf. Useful for:
- Computing rewards for individual image pairs.
- Debugging embedding distances.
- Preparing expert reference images.

```python
calc = DINORewardCalculator()
reward = calc.compute_handoff_reward(current_img, expert_ref_img, temperature=1.0)
```

## 4. Configuration

### 4.1 Training Config (YAML)

Example: `examples/embodiment/config/maniskill_ppo_mlp_dinov3_reward.yaml`

Key sections:

```yaml
reward:
  use_reward_model: True
  group_name: "RewardGroup"
  reward_mode: "terminal"      # "terminal" or "per_step"
  continuous_reward: True      # Return raw similarity scores
  reward_threshold: 0.5        # Only used if continuous_reward=False
  reward_weight: 1.0           # Weight of learned reward
  env_reward_weight: 0.0       # Weight of environment reward
  enable_offload: False        # Move model to CPU between steps

  model:
    model_path: null
    model_type: "dinov3"
    model_name: "facebook/dinov3-vitl16-pretrain-lvd1689m"
    expert_ref_image_path: "/path/to/expert_ref.png"
    temperature: 1.0
    image_size: [3, 224, 224]   # Compatibility field
    normalize: false            # DINOv3 processor handles normalization
    precision: "fp32"           # Use fp32 for numerical stability
```

### 4.2 Reward Mode

- **`terminal`**: Computes reward only at episode termination using the final observation. The reward is scattered back to the terminal step. Sparse but stable.
- **`per_step`**: Computes reward at every step using the current observation. Denser but may be noisier for intermediate states that don't resemble the expert reference.

For manipulation tasks with clear goal states, `terminal` is typically preferred.

### 4.3 Model Variants

Supported checkpoints (all from `facebook/` on HuggingFace):

| Model | Parameters | Embedding Dim | Speed | Memory |
|---|---|---|---|---|
| `dinov3-vits16` | ~22M | 384 | Fastest | ~0.5 GB |
| `dinov3-vitb16` | ~86M | 768 | Fast | ~1.3 GB |
| `dinov3-vitl16` | ~304M | 1024 | Medium | ~2.5 GB |

Larger models produce more semantically meaningful embeddings but consume more VRAM. For a dedicated reward worker GPU, any variant is feasible.

### 4.4 Local Checkpoints

`model_name` accepts either:
- HuggingFace model ID (downloads on first use): `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Local directory path (for offline/air-gapped servers): `/root/autodl-tmp/checkpoints/dinov3/...`

## 5. Testing Hierarchy

### 5.1 L1: Unit Test

**File**: `tests/unit_tests/test_dinov3_reward_model.py`

Tests model registration, instantiation, forward pass, and `compute_reward()`.

Run:
```bash
cd RLinf/tests/unit_tests
python test_dinov3_reward_model.py
```

Validates:
- Model loads successfully from local checkpoint.
- Forward pass handles NHWC uint8 and NCHW float inputs.
- Output probabilities are in \([0, 1]\).
- Identical images to expert ref give reward \(\approx 1.0\).

### 5.2 L2: Smoke Test

**Config**: `examples/embodiment/config/maniskill_ppo_mlp_dinov3_reward_smoke.yaml`

A minimal training loop (5 steps, 8 envs) that verifies:
- Env worker starts and produces observations with images.
- Reward worker receives images and returns rewards.
- Actor worker receives rewards and performs at least one optimization step.
- No crashes in the distributed pipeline.

Run:
```bash
cd RLinf/examples/embodiment
python train_embodied_agent.py --config-name=maniskill_ppo_mlp_dinov3_reward_smoke
```

### 5.3 L3: Task Integration Test

**Config**: `examples/embodiment/config/robotwin_stack_block_three_ppo_mlp_dinov3_reward_smoke.yaml`

Validates the reward model on a real robot manipulation task (Robotwin). Requires:
- Task assets and seeds on the server.
- Expert reference image for the specific task.
- Correct `obs_dim` for the MLP policy.

Run:
```bash
cd RLinf/examples/embodiment
python train_embodied_agent.py --config-name=robotwin_stack_block_three_ppo_mlp_dinov3_reward_smoke
```

## 6. Expert Reference Image

The expert reference image is the **single most important hyperparameter** for this reward model.

### 6.1 Requirements

- Format: PNG or JPEG.
- Content: A single camera frame showing the **desired final state** of the task.
- Resolution: Should match the env's image output (typically 224x224 or 256x256).
- Camera: Should ideally match the env's `main_image` camera perspective.

### 6.2 How to Obtain

1. **From expert demonstrations**: Extract the final frame from a successful trajectory.
2. **From the environment**: Run the env with a scripted/heuristic policy, save a frame from a successful episode.
3. **Manual capture**: Set up the scene in the simulator and take a screenshot.

### 6.3 Best Practices

- Use a **clean background** if possible (reduces irrelevant features).
- Ensure the **goal object configuration is clearly visible**.
- Avoid **occlusions** (e.g., robot arm blocking the goal state).
- If the task has multiple valid final states, pick the **most representative** one.

### 6.4 Validation

Before full training, validate the reference image:

```python
from examples.reward.dino_reward_calculator import DINORewardCalculator
from PIL import Image

calc = DINORewardCalculator()
ref = Image.open("/path/to/expert_ref.png")
cur = Image.open("/path/to/random_state.png")

r_same = calc.compute_handoff_reward(ref, ref)
r_diff = calc.compute_handoff_reward(cur, ref)
print(f"Same: {r_same:.4f}, Diff: {r_diff:.4f}, Gap: {r_same - r_diff:.4f}")
```

A good reference image should give:
- \(R(\text{ref}, \text{ref}) \approx 1.0\)
- \(R(\text{random}, \text{ref}) \ll 1.0\) (ideally \(< 0.5\))
- Large gap between success and failure states.

## 7. Known Limitations

1. **Viewpoint sensitivity**: If the expert ref and env observations come from different camera angles, embeddings may differ even for the same scene state. Ensure camera consistency.

2. **Task ambiguity**: For tasks where the goal is not visually distinctive (e.g., "push the mug 5cm to the left"), DINOv3 embeddings may not capture fine-grained spatial differences well.

3. **Partial observability**: If the goal state involves occluded objects or requires information not visible in the main camera, the reward will be unreliable.

4. **No temporal information**: The reward is computed from a single frame. It cannot distinguish between "about to succeed" and "already succeeded" if both frames look similar.

5. **Frozen backbone**: DINOv3 is not fine-tuned for the specific task or robot. For tasks very different from its training distribution (web-scale images), embeddings may be less discriminative.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Can't load image processor" | Wrong model path or no HF access | Check `model_name` path; verify files exist |
| Reward always ~1.0 | Expert ref too generic | Use a more specific goal-state image |
| Reward always ~0.0 | Temperature too low or embeddings broken | Increase `temperature`; check normalization |
| OOM on reward worker | Model too large for GPU | Use `vitb16` or `vits16`; enable `enable_offload` |
| "images" key missing in reward input | Env not configured for images | Check env returns `main_images`; verify `obs_mode` includes RGB |
| All rewards identical | Input images are identical | Check env produces diverse observations |

## 9. File Reference

| File | Purpose |
|---|---|
| `rlinf/models/embodiment/reward/dinov3_reward_model.py` | Core reward model implementation |
| `rlinf/models/embodiment/reward/__init__.py` | Model registry registration |
| `rlinf/workers/reward/reward_worker.py` | Distributed reward worker (receives obs, returns rewards) |
| `rlinf/workers/env/env_worker.py` | Env worker (extracts `main_images`, sends to reward worker) |
| `examples/reward/dino_reward_calculator.py` | Standalone calculator for local testing |
| `examples/embodiment/config/maniskill_ppo_mlp_dinov3_reward.yaml` | Full ManiSkill training config |
| `examples/embodiment/config/maniskill_ppo_mlp_dinov3_reward_smoke.yaml` | ManiSkill smoke test config |
| `examples/embodiment/config/robotwin_stack_block_three_ppo_mlp_dinov3_reward_smoke.yaml` | Robotwin integration test config |
| `tests/unit_tests/test_dinov3_reward_model.py` | Unit tests for model correctness |
