# Handoff-Aware RL Post-Training: RLinf Adaptation Plan

## Context

The goal is to implement "Handoff-Aware RL Post-Training for Long-Horizon Manipulation" within the RLinf framework. The core problem: in sequential manipulation tasks (e.g., CALVIN's 5-task chains), subtask transitions fail because the terminal state of subtask $G_i$ doesn't align with the start conditions of $G_{i+1}$. The solution: teach a VLA to shape its terminal states to be handoff-friendly via RL post-training with a specialized handoff reward.

RLinf already provides nearly all infrastructure needed — PPO, GAE, ValueHead, LoRA, FSDP distributed training, CALVIN environment with sequential subtasks, SFT runner, and a clean registry-based algorithm system. The adaptation primarily adds new components through existing extension points rather than rewriting core code.

---

## Phase 1: Data Structures — Extend Trajectory Pipeline

**File**: `rlinf/data/embodied_io_struct.py`

### 1.1 Extend `ChunkStepResult` (line ~244)
Add optional fields:
```python
prev_alpha: torch.Tensor = None            # [B, 1] — α-head sigmoid output
prev_handoff_values: torch.Tensor = None    # [B, 1] — handoff value head output
subtask_boundaries: torch.Tensor = None     # [B, 1] — bool, True when subtask just completed
```
Add `.cpu().contiguous()` handling in `__post_init__` following existing pattern (lines 257-275).

### 1.2 Extend `Trajectory` (line ~278)
Add matching tensor fields:
```python
prev_alpha: torch.Tensor = None
prev_handoff_values: torch.Tensor = None
subtask_boundaries: torch.Tensor = None
```

### 1.3 Extend `EmbodiedRolloutResult` (line ~396)
Add `list[torch.Tensor]` fields for each new field. Update:
- `append_step_result()` (line ~432): append new fields following the `prev_values` pattern
- `to_trajectory()` (line ~495): stack new fields with `torch.stack(...).cpu().contiguous()`

### 1.4 Extend `EnvOutput` (line ~31)
Add:
```python
subtask_boundaries: Optional[torch.Tensor] = None  # [B]
```
Update `prepare_observations()` to propagate `next_task_descriptions` from obs dict.

**Why these changes**: The trajectory pipeline is the backbone of all data flow in RLinf. All new per-step signals (α, handoff value, subtask boundary) must travel through this pipeline from rollout worker → actor worker. Using optional fields with `None` defaults ensures zero impact on existing training configs.

---

## Phase 2: Model Heads — α-Head and Handoff Value Head

### 2.1 OpenPI Model: `rlinf/models/embodiment/openpi/openpi_action_model.py`

**Extend `OpenPi0Config`** (line ~37) with:
```python
add_alpha_head: bool = False
add_handoff_value_head: bool = False
```

**In `__init__`** (after value_head block, line ~157), add:
```python
if self.config.add_alpha_head:
    self.alpha_head = ValueHead(
        input_dim=proj_width,  # 1024 or 2048 depending on value_after_vlm
        hidden_sizes=(256, 64),
        output_dim=1,
        activation="relu",
        bias_last=True,
    )

if self.config.add_handoff_value_head:
    self.handoff_value_head = ValueHead(
        input_dim=proj_width,
        hidden_sizes=value_head_hidden_sizes,  # same as value_head
        output_dim=1,
        activation=value_head_activation,
        bias_last=True,
    )
```

**In `sample_actions`** (where `suffix_out_value` is computed for value_head), add:
```python
if hasattr(self, "alpha_head"):
    alpha_t = torch.sigmoid(self.alpha_head(suffix_out_value))
else:
    alpha_t = None

if hasattr(self, "handoff_value_head"):
    handoff_value_t = self.handoff_value_head(suffix_out_value)
else:
    handoff_value_t = None
```

**In `predict_action_batch`** result dict, add:
```python
result["prev_alpha"] = alpha_t           # [B, 1] or None
result["prev_handoff_values"] = handoff_value_t  # [B, 1] or None
```

**In `default_forward`** (training forward), compute alpha and handoff value the same way, returning them for loss computation.

### 2.2 Model Factory: `rlinf/models/__init__.py`

Extend the LoRA trainable-params section (line ~87):
```python
for head_name in ("value_head", "alpha_head", "handoff_value_head"):
    if hasattr(model, head_name):
        for param in getattr(model, head_name).parameters():
            param.requires_grad = True
```

**Why separate heads**: The existing `value_head` estimates V(s) for the current subtask's discounted return. The `handoff_value_head` estimates a different quantity — reachability toward the next subtask's start distribution. They have different optimization targets and must not share parameters. The `alpha_head` is very lightweight (~10K params) and predicts the transition phase.

---

## Phase 3: Environment — Sequential Task Support

### 3.1 CALVIN Environment Enhancement: `rlinf/envs/calvin/calvin_gym_env.py`

CALVIN already implements sequential subtask logic (5-task chains) with `_check_subtask_success()` (line 458), `_reset_current_task()` (line 473), `task_sequence` tracking, and `current_task_idx`. The adaptation requires:

**Add `next_task_descriptions` tracking** in `_init_task_info()` (line 133):
```python
self.next_task_descriptions = [None] * self.num_envs
```

**In `_get_task_info()`** (line 203), populate next task:
```python
if len(self.task_sequence[i]) > 1:
    next_task = self.task_sequence[i][1]
    self.next_task_descriptions[i] = self.task_suite.get_task_descriptions(next_task)
else:
    self.next_task_descriptions[i] = "none"
```

**In `_reset_current_task()`** (line 473), when advancing subtask:
```python
# After updating current_task and task_descriptions
next_idx = self.current_task_idx[env_id] + 1
if next_idx < len(self.task_sequence[env_id]):
    next_task = self.task_sequence[env_id][next_idx]
    self.next_task_descriptions[env_id] = self.task_suite.get_task_descriptions(next_task)
else:
    self.next_task_descriptions[env_id] = "none"
```

**In `_wrap_obs()`** (line 272), add to obs dict:
```python
obs["next_task_descriptions"] = list(self.next_task_descriptions)
```

**In `step()`** (line 346), expose subtask boundaries:
The existing `subtask_success` array (line 352) is exactly the subtask boundary signal. Add to return:
```python
# After _check_subtask_success
infos["subtask_boundaries"] = to_tensor(subtask_success)
```

**In `chunk_step()`** (line 378), collect subtask boundaries across chunk steps (any boundary in the chunk = True).

### 3.2 Dual-Task Prompt Construction

In the rollout worker (see Phase 5), construct the prompt:
```
Current task: {task_descriptions[i]}
Next task: {next_task_descriptions[i]}
Transition phase: {alpha_bin_text}
```

Where `alpha_bin_text` is derived from the predicted `alpha_t`:
- [0.0, 0.2): "focus on current task"
- [0.2, 0.5): "begin preparing for transition"
- [0.5, 0.8): "actively transitioning"
- [0.8, 1.0]: "prioritize next task readiness"

This replaces the single `task_descriptions` field currently used. The prompt can be constructed in `MultiStepRolloutWorker.predict()` before calling `predict_action_batch`, or in a preprocessing step on `env_obs["task_descriptions"]`.

---

## Phase 4: Handoff Reward

### 4.1 Expert Start-State Library

**New file**: `rlinf/algorithms/rewards/handoff_reward.py`

```python
class ExpertStartStateLibrary:
    """Cached VLM embeddings from expert demo start states, per subtask."""
    
    def __init__(self, embeddings_path: str, device: str = "cpu"):
        # Load: Dict[subtask_name -> Tensor[N_samples, D_embed]]
        self.embeddings = torch.load(embeddings_path, map_location=device)
        self.embed_dim = next(iter(self.embeddings.values())).shape[-1]
    
    def get_anchors(self, subtask_name: str) -> torch.Tensor:
        """Return all anchor embeddings for a subtask. [N, D]"""
        return self.embeddings.get(subtask_name, torch.zeros(1, self.embed_dim))
```

### 4.2 Reach Function and Handoff Reward

```python
def compute_reach(
    z_t: torch.Tensor,                # [B, D] current visual embedding
    anchor_embeddings: torch.Tensor,   # [N, D] expert start-state embeddings
    tau: float = 0.5,                  # softmin temperature
    beta: float = 1.0,                 # reach temperature
) -> torch.Tensor:
    """Compute handoff reachability. Returns [B] in (0, 1]."""
    # z_t: [B, D], anchor: [N, D] -> dists: [B, N]
    dists_sq = torch.cdist(z_t, anchor_embeddings, p=2).pow(2)  # [B, N]
    softmin_dist = -tau * torch.logsumexp(-dists_sq / tau, dim=-1)  # [B]
    reach = torch.exp(-softmin_dist / beta)  # [B]
    return reach

def compute_handoff_reward(
    reach_cur: torch.Tensor,   # [B] Reach at current step
    reach_prev: torch.Tensor,  # [B] Reach at previous step
) -> torch.Tensor:
    """Incremental handoff reward: did this step improve reachability?"""
    return reach_cur - reach_prev  # [B]
```

### 4.3 Reward Integration Point

The handoff reward is computed in the **rollout worker** during trajectory collection, because:
1. The visual embedding `z_t` comes from the VLA's forward pass (available in rollout worker)
2. No extra communication round needed
3. The reward is added to `rewards` in `ChunkStepResult` before it enters the trajectory pipeline

The visual embedding extraction: In OpenPI, the `suffix_out_value` (the hidden state used for value_head) is the appropriate representation. It can be returned alongside other results.

### 4.4 Precomputation Script

**New file**: `toolkits/handoff/precompute_start_embeddings.py`

Standalone script that:
1. Loads the frozen VLA model
2. Iterates through expert demonstration datasets
3. For each subtask, extracts the visual embedding at the first frame
4. Saves `Dict[subtask_name -> Tensor[N, D]]` as a .pt file

### 4.5 Register Reward

In `rlinf/algorithms/rewards/__init__.py`, add:
```python
from rlinf.algorithms.rewards.handoff_reward import HandoffReward
register_reward("handoff", HandoffReward)
```

---

## Phase 5: Rollout Worker — Handoff-Aware Trajectory Collection

**File**: `rlinf/workers/rollout/hf/huggingface_worker.py`

### 5.1 Initialization

In `__init__` or `init_worker()`, load the expert start-state library:
```python
handoff_cfg = self.cfg.algorithm.get("handoff", None)
if handoff_cfg and handoff_cfg.enabled:
    self.expert_library = ExpertStartStateLibrary(handoff_cfg.expert_embeddings_path)
    self.handoff_lambda = handoff_cfg.reward_scale
    self.reach_prev = {}  # per-stage tracking of previous Reach values
else:
    self.expert_library = None
```

### 5.2 Modified `generate_one_epoch()` (line ~315)

In the inner loop, after `actions, result = self.predict(env_output["obs"])`:

```python
# Extract new head outputs
prev_alpha = result.get("prev_alpha", None)
prev_handoff_values = result.get("prev_handoff_values", None)

# Extract subtask boundaries from env
subtask_boundaries = env_output.get("subtask_boundaries", None)

# Compute handoff reward if enabled
if self.expert_library is not None and env_output["obs"].get("next_task_descriptions"):
    visual_emb = result.get("visual_embeddings")  # [B, D]
    next_tasks = env_output["obs"]["next_task_descriptions"]
    
    # Compute Reach for each env
    reach_cur = self._compute_batch_reach(visual_emb, next_tasks)
    reach_prev = self.reach_prev.get(stage_id, torch.zeros_like(reach_cur))
    handoff_reward = (reach_cur - reach_prev) * self.handoff_lambda
    
    # Gate by alpha: handoff reward only active near transitions
    if prev_alpha is not None:
        handoff_reward = handoff_reward * prev_alpha.squeeze(-1)
    
    # Add to env reward
    if rewards is not None:
        rewards = rewards + handoff_reward.unsqueeze(-1)
    
    self.reach_prev[stage_id] = reach_cur
```

### 5.3 Extended ChunkStepResult Construction

```python
chunk_step_result = ChunkStepResult(
    actions=result["forward_inputs"].get("action", None),
    dones=dones,
    rewards=rewards,  # now includes handoff reward
    truncations=env_output["truncations"],
    terminations=env_output["terminations"],
    prev_logprobs=result["prev_logprobs"],
    prev_values=result["prev_values"],
    prev_alpha=prev_alpha,                    # NEW
    prev_handoff_values=prev_handoff_values,  # NEW
    subtask_boundaries=subtask_boundaries,    # NEW
    forward_inputs=result["forward_inputs"],
)
```

### 5.4 Dual-Task Prompt

Before calling `self.predict()`, modify `env_obs["task_descriptions"]` to include next task:
```python
if self.expert_library is not None:
    task_descs = env_output["obs"]["task_descriptions"]
    next_descs = env_output["obs"].get("next_task_descriptions", ["none"] * len(task_descs))
    # Replace task_descriptions with dual prompt
    env_output["obs"]["task_descriptions"] = [
        f"Current task: {cur}. Next task: {nxt}. Transition phase: focus on current task"
        for cur, nxt in zip(task_descs, next_descs)
    ]
```

In later iterations (once α-head is warmed up), use α predictions from the previous step to set the transition phase text.

---

## Phase 6: Algorithms — Handoff-Aware GAE and Loss

### 6.1 Handoff GAE

**File**: `rlinf/algorithms/advantages.py`

Register a new advantage function:
```python
@register_advantage("handoff_gae")
def compute_handoff_gae(
    rewards, gamma, gae_lambda, values,
    handoff_values=None,        # [T+1, B] from handoff_value_head
    subtask_boundaries=None,    # [T, B] bool
    normalize_advantages=True,
    loss_mask=None, dones=None,
    **kwargs,
):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    for step in reversed(range(T)):
        # Subtask boundary acts like an episode boundary for GAE
        is_boundary = subtask_boundaries[step] if subtask_boundaries is not None else False
        is_terminal = dones[step + 1] | is_boundary

        # At boundary, bootstrap with handoff value
        if handoff_values is not None and subtask_boundaries is not None:
            next_val = torch.where(
                subtask_boundaries[step].bool(),
                handoff_values[step + 1],
                values[step + 1],
            )
        else:
            next_val = values[step + 1]

        delta = rewards[step] + gamma * next_val * (~is_terminal) - values[step]
        gae = delta + gamma * gae_lambda * (~is_terminal) * gae
        returns[step] = gae + values[step]

    advantages = returns - values[:-1]
    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    return advantages, returns
```

**Key difference from standard GAE**: At subtask boundaries, (1) the GAE trace resets (no credit propagation across subtasks), and (2) the bootstrap value switches from task-value to handoff-value.

### 6.2 Preprocessing Support

**File**: `rlinf/algorithms/utils.py`

In `preprocess_embodied_advantages_inputs()` (line ~67), add handling for new fields:
```python
# After existing dones/values processing (line ~119)
if kwargs.get("subtask_boundaries") is not None:
    sb = kwargs["subtask_boundaries"]  # [num_chunk, bsz, chunk_size]
    if kwargs["reward_type"] == "chunk_level":
        sb = sb.max(dim=-1, keepdim=True)[0]
    sb = sb.transpose(1, 2).reshape(n_steps, bsz)
    kwargs["subtask_boundaries"] = sb

if kwargs.get("prev_handoff_values") is not None and kwargs["adv_type"] == "handoff_gae":
    hv = kwargs["prev_handoff_values"]  # same shape as values
    flattened_hv = hv.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, bsz)
    kwargs["handoff_values"] = flattened_hv[:n_steps + 1]
```

### 6.3 Handoff Loss Function

**File**: `rlinf/algorithms/losses.py`

```python
@register_policy_loss("handoff_actor_critic")
def compute_handoff_ppo_loss(**kwargs):
    """PPO actor+critic + α supervision + handoff value loss."""
    # Standard PPO
    actor_loss, actor_metrics = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics = compute_ppo_critic_loss(**kwargs)
    
    metrics = {}
    metrics.update(actor_metrics)
    metrics.update(critic_metrics)
    loss = actor_loss + critic_loss
    
    # α-head supervision loss
    alpha_preds = kwargs.get("alpha_preds")     # [B, 1]
    alpha_targets = kwargs.get("alpha_targets")  # [B, 1]
    alpha_weight = kwargs.get("alpha_loss_weight", 1.0)
    if alpha_preds is not None and alpha_targets is not None:
        alpha_loss = F.mse_loss(alpha_preds, alpha_targets)
        loss = loss + alpha_weight * alpha_loss
        metrics["handoff/alpha_loss"] = alpha_loss.detach()
    
    # Handoff value loss (reuse PPO critic loss)
    handoff_vals = kwargs.get("handoff_values")
    handoff_returns = kwargs.get("handoff_returns")
    handoff_prev_vals = kwargs.get("prev_handoff_values")
    handoff_weight = kwargs.get("handoff_value_loss_weight", 0.5)
    if handoff_vals is not None and handoff_returns is not None:
        hv_loss, hv_metrics = compute_ppo_critic_loss(
            values=handoff_vals,
            returns=handoff_returns,
            prev_values=handoff_prev_vals,
            value_clip=kwargs["value_clip"],
            huber_delta=kwargs["huber_delta"],
            loss_mask=kwargs.get("loss_mask"),
        )
        loss = loss + handoff_weight * hv_loss
        metrics["handoff/value_loss"] = hv_loss.detach()
    
    return loss, metrics
```

---

## Phase 7: Actor Worker — Forward New Fields

**File**: `rlinf/workers/actor/fsdp_actor_worker.py`

### 7.1 `compute_advantages_and_returns()`

Ensure the batch dict passed to `calculate_adv_and_returns()` includes:
- `subtask_boundaries` from trajectory
- `prev_handoff_values` from trajectory (renamed to `handoff_values` after preprocessing)

These are already in the trajectory tensors from Phase 1 changes. The existing code path in `EmbodiedFSDPActor` converts trajectories to batch dicts via `convert_trajectories_to_batch()`, which already handles arbitrary tensor fields.

### 7.2 `run_training()` Forward Pass

In the training forward, the model must return `alpha_preds` and `handoff_values`. These are added to the loss kwargs:
```python
loss_kwargs["alpha_preds"] = forward_result["alpha_preds"]
loss_kwargs["alpha_targets"] = batch["alpha_targets"]  # from trajectory or computed
loss_kwargs["handoff_values"] = forward_result["handoff_values"]
loss_kwargs["handoff_returns"] = batch["handoff_returns"]
loss_kwargs["prev_handoff_values"] = batch["prev_handoff_values"]
```

### 7.3 α-Target Computation

Alpha targets for RL rollout data: computed from subtask boundaries using a ramp-up schedule.
```python
# In advantage computation or as preprocessing:
# For each episode, α ramps from 0 to 1 over the last w steps before subtask completion
# This can be computed post-hoc from subtask_boundaries tensor
```

For expert demonstration data (mixed in for α supervision): precomputed offline with the linear ramp-up formula from the plan.

---

## Phase 8: Training Configuration

### 8.1 Example Config

**New file**: `examples/embodiment/config/calvin_handoff_ppo_openpi.yaml`

```yaml
defaults:
  - env: calvin_abcd
  - model: openpi_calvin
  - backend: fsdp

runner:
  task_type: embodied
  max_epochs: 5000
  val_check_interval: 100
  save_interval: 500

algorithm:
  adv_type: handoff_gae
  loss_type: handoff_actor_critic
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  
  handoff:
    enabled: true
    expert_embeddings_path: "/path/to/calvin_expert_start_embeddings.pt"
    reward_scale: 10.0           # λ_h
    alpha_loss_weight: 1.0       # λ_α
    handoff_value_loss_weight: 0.5  # λ_V
    smoothness_penalty: 0.1      # λ_s
    alpha_threshold: 0.2         # smoothness activation threshold
    reach_tau: 0.5               # softmin temperature
    reach_beta: 1.0              # reach temperature
    transition_window: 20        # w steps for α ramp-up
    reward_warmup_steps: 100     # warmup λ_h from 1.0 to target

actor:
  model:
    add_value_head: true
    add_alpha_head: true
    add_handoff_value_head: true
    is_lora: true
    lora_rank: 32
```

### 8.2 Two-Stage Training

**Stage 1 (SFT)**: Use existing `SFTRunner` with:
- Dual-task prompt format in training data
- α-head warmup: all α targets = 0 (no next task during SFT)
- Config: `algorithm.loss_type: sft` with `add_alpha_head: true`

**Stage 2 (RL)**: Use `EmbodiedRunner` with:
- Config above (handoff_gae + handoff_actor_critic)
- `resume_dir` pointing to Stage 1 checkpoint
- KL constraint against SFT checkpoint via `kl_beta` config

---

## Phase 9: Verification Plan

### 9.1 Unit Tests
1. **Data pipeline round-trip**: Create a `ChunkStepResult` with all new fields → append to `EmbodiedRolloutResult` → convert to `Trajectory` → verify shapes
2. **Handoff GAE**: Compare `handoff_gae` with `gae` on trajectories without subtask boundaries (should be identical)
3. **Reach function**: Verify `compute_reach` gives ~1.0 for embeddings near anchors, ~0 for far ones
4. **Loss function**: Verify `handoff_actor_critic` reduces to `actor_critic` when α_preds and handoff_values are None

### 9.2 Integration Tests
1. **CALVIN env**: Run CALVIN with `next_task_descriptions` exposed, verify correct subtask progression
2. **Rollout with handoff reward**: Run one rollout epoch with handoff reward enabled, verify rewards have handoff component
3. **End-to-end training**: Run 10 steps of Stage 2 training, verify loss decreases and all metrics logged

### 9.3 Ablation Configs
Prepare config variants for:
- No handoff reward (`handoff.reward_scale: 0`)
- Fixed α schedule (`add_alpha_head: false`, use linear ramp)
- No α gating (handoff reward not multiplied by α)
- No KL constraint (`kl_beta: 0`)

---

## Summary of Files

### New Files
| File | Purpose |
|------|---------|
| `rlinf/algorithms/rewards/handoff_reward.py` | ExpertStartStateLibrary, Reach function, HandoffReward class |
| `toolkits/handoff/precompute_start_embeddings.py` | Offline embedding precomputation script |
| `examples/embodiment/config/calvin_handoff_ppo_openpi.yaml` | Training config |
| `examples/embodiment/train_handoff.py` | Entry point (thin wrapper) |

### Modified Files
| File | Changes |
|------|---------|
| `rlinf/data/embodied_io_struct.py` | Add prev_alpha, prev_handoff_values, subtask_boundaries to ChunkStepResult/Trajectory/EmbodiedRolloutResult/EnvOutput |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | Add alpha_head, handoff_value_head; extend forward and predict |
| `rlinf/models/__init__.py` | Ensure new heads are trainable with LoRA (line ~87) |
| `rlinf/algorithms/advantages.py` | Register `handoff_gae` |
| `rlinf/algorithms/losses.py` | Register `handoff_actor_critic` |
| `rlinf/algorithms/utils.py` | Extend `preprocess_embodied_advantages_inputs` for new fields |
| `rlinf/algorithms/rewards/__init__.py` | Register handoff reward |
| `rlinf/envs/calvin/calvin_gym_env.py` | Expose next_task_descriptions, subtask_boundaries |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | Handoff reward computation, dual-task prompt, new ChunkStepResult fields |
| `rlinf/workers/actor/fsdp_actor_worker.py` | Forward new fields through advantage/loss computation |
