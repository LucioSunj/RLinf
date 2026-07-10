# 基于 RLinf + OpenVLA-OFT 的 Handoff-Aware Long-Horizon Manipulation 实验方案

## 0. RLinf 现有代码框架与本实验的对应关系

下面这张图先把当前 `RLinf` 中与本实验直接相关的代码链路画出来，并标出建议新增的位置。

```text
Run Script
  bash examples/embodiment/run_embodiment.sh <config_name>
    |
    v
examples/embodiment/train_embodied_agent.py
    |
    v
rlinf/runners/embodied_runner.py
    |
    +--> Env Worker Group
    |      rlinf/workers/env/env_worker.py
    |         |
    |         +--> rlinf/envs/__init__.py
    |         |      |
    |         |      +--> rlinf/envs/maniskill/maniskill_env.py
    |         |      |
    |         |      +--> [NEW] rlinf/envs/maniskill/handoff_maniskill_env.py
    |         |             - 维护子任务队列 [G_cur, G_next]
    |         |             - 输出结构化 task_descriptions
    |         |             - 计算 r_done / r_handoff / r_smooth
    |         |             - 在 info 中记录 task_complete / handoff 指标
    |         |
    |         +--> env.step() / env.chunk_step()
    |                |
    |                +--> reward 在这里产出，而不是 RewardWorker
    |
    +--> Rollout Worker Group
    |      rlinf/workers/rollout/hf/huggingface_worker.py
    |         |
    |         +--> 调用模型 predict(...)
    |         +--> 收集 prev_logprobs / prev_values / forward_inputs
    |         +--> 把 trajectory 发给 actor
    |
    +--> Actor Worker Group
           rlinf/workers/actor/fsdp_actor_worker.py
              |
              +--> rlinf/models/__init__.py
              |      |
              |      +--> rlinf/models/embodiment/openvla_oft/__init__.py
              |      |      |
              |      |      +--> rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py
              |      |             - 接收 env_obs["task_descriptions"] 作为语言输入
              |      |             - 输出动作 token / logprob
              |      |             - 使用现有 value_head 做 PPO critic
              |      |             - 支持 LoRA
              |      |
              |      +--> peft LoRA 注入逻辑在 rlinf/models/__init__.py
              |
              +--> compute_advantages_and_returns()
              +--> run_training()
                     - PPO loss
                     - value loss
                     - optional KL
                     - critic warmup

Configs
  examples/embodiment/config/model/openvla_oft.yaml
  examples/embodiment/config/maniskill_ppo_openvlaoft.yaml
  [NEW] examples/embodiment/config/env/maniskill_handoff_template.yaml
  [NEW] examples/embodiment/config/maniskill_ppo_openvlaoft_handoff.yaml
```

### 0.1 当前 RLinf 代码对本实验的硬约束

1. `OpenVLA-OFT` 在当前 `RLinf` 中走的是 `actor_critic` / PPO 路线，不是 `embodied_sac`。
   原因是 `openvla_oft_action_model.py` 当前只实现 `ForwardType.DEFAULT`，没有 `SAC` / `SAC_Q` 分支。

2. embodied 训练的 reward 默认由环境侧产生。
   也就是说，本实验的 handoff reward 应优先落在 env / env wrapper，而不是 `RewardWorker`。

3. `OpenVLA-OFT` 已经原生接收 `task_descriptions` 作为 prompt。
   因此双任务 prompt 不需要新增门控注意力，也不需要重写 rollout 框架。

4. `OpenVLA-OFT` 的 PPO critic 已经现成支持。
   第一版实验不需要先发明新的 critic 训练框架。

5. `RLinf` 当前的 VLA SFT worker 只明确支持 `OpenPI`，不支持 `OpenVLA-OFT`。
   因此本方案默认 Stage 1 使用“外部已有的 OpenVLA-OFT SFT checkpoint”，而不是假设在 RLinf 内完成 OpenVLA-OFT 的 SFT。

---

## 1. 实验目标与研究问题

### 1.1 核心目标

把原本偏 `Decision Transformer / Reinformer` 风格的 long-horizon handoff 方案，改造成：

- backbone 使用 `OpenVLA-OFT`
- 训练框架使用 `RLinf`
- 优化方式使用 `PPO` 风格 RL post-training
- 关键创新点保留在 `handoff-aware reward + dual-task prompt + transition-aware policy shaping`

### 1.2 第一性问题

本实验第一阶段真正要回答的问题只有一个：

> 在不重写 RL 框架、不先发明新架构的前提下，仅通过 `dual-task prompt + handoff reward + PPO post-training`，是否能显著提升 long-horizon chain success rate？

只要这个问题没有被验证，就不应该第一版同时引入：

- learned `alpha-head`
- 新的 handoff-specific critic
- 纯 offline RL
- 新的 policy architecture
- 自定义轻量 RL 框架

### 1.3 本方案的研究假设

1. 预训练 / SFT 的 `OpenVLA-OFT` 已经具备单子任务 instruction following 能力。
2. long-horizon 性能下降的主要来源是 handoff 失败，而不是每个子任务内部的局部控制能力不足。
3. 在 prompt 中显式注入 `G_cur + G_next + transition phase`，再叠加 handoff reward，可以让 policy 在当前子任务末段主动塑造更利于下一个子任务启动的状态。
4. 这个能力可以通过 `RL post-training` 学到，不需要引入 RTG 条件建模。

---

## 2. 总体实验路线

### 2.1 总体策略

本实验建议分三版推进：

1. `V1: MVP`
   不改模型结构，只改环境、prompt、reward、配置。

2. `V2: Alpha-Head`
   在 `OpenVLA-OFT` 上增加一个轻量 `alpha-head`，让 transition phase 从硬编码 schedule 变成可学习预测。

3. `V3: Dedicated Handoff Critic`
   只有在 V1/V2 明确证明 handoff reward 有效后，才考虑把现有 PPO value head 改成更显式的 handoff critic。

### 2.2 为什么第一版先做 PPO 而不是 SAC / 离线 RL

对 `OpenVLA-OFT` 而言，当前 `RLinf` 最成熟、最贴合代码现状的主线是 PPO：

- `maniskill_ppo_openvlaoft.yaml` 已经是现成 recipe
- `OpenVLA-OFT` 已有 `value_head`
- `actor_critic + GAE + PPO clip + KL + critic warmup` 都已经打通

相反，以下路线都不适合作为第一版主线：

- `OpenVLA-OFT + embodied_sac`
  当前模型接口没有现成支持。
- 纯静态 offline RL
  当前 `RLinf` embodied 主线仍然是 env/rollout 驱动。
- 自己写新框架
  工程时间会被 rollout、logging、checkpoint、distributed 细节吞掉。

因此，本方案的主线是：

```text
外部已有 OpenVLA-OFT SFT checkpoint
    ->
RLinf + OpenVLA-OFT + PPO
    ->
long-horizon env / wrapper
    ->
handoff-aware reward
    ->
逐步加入 alpha-head 与更强 critic
```

---

## 3. 实验版本设计

## 3.1 V1: 最小可验证版本

### 3.1.1 目标

用最小改动验证：

- `dual-task prompt` 是否有效
- `handoff reward` 是否有效
- `RL post-training` 是否优于“只靠 SFT + 单任务 reward”

### 3.1.2 V1 不改的部分

V1 中以下部分全部复用现有 `RLinf`：

- runner
- actor / rollout / env worker 通信
- OpenVLA-OFT 主干
- value head
- PPO loss
- GAE
- checkpoint
- tensorboard / logging

### 3.1.3 V1 只改的部分

V1 只动四个点：

1. 新建 long-horizon / handoff-aware 的 ManiSkill env wrapper
2. 将 `task_descriptions` 改成双任务结构化 prompt
3. 在 env 侧计算 handoff reward
4. 新建一份 handoff 专用 config

### 3.1.4 V1 的模型定义

V1 仍然是标准 `OpenVLA-OFT + value head + LoRA`：

- action head：沿用现有实现
- value head：沿用现有实现
- no alpha-head
- no new attention module
- no RTG head

也就是说，V1 的“算法创新”完全落在：

- prompt 设计
- reward 设计
- 环境状态机设计

而不是新的 Transformer 结构。

## 3.2 V2: 加入 Alpha-Head

### 3.2.1 动机

当 V1 验证 handoff reward 确实有帮助之后，再把 transition phase 从“手工 schedule”升级为“模型预测”。

### 3.2.2 实现方式

在 `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py` 中，仿照现有 `value_head` 增加一个轻量 `alpha_head`：

- 输入：与 value head 相同的 `hidden_features`
- 输出：标量 `alpha_hat in [0, 1]`
- 用途：
  - 训练期做 auxiliary supervision
  - 推理期将连续值离散为 phase token
  - 记录 transition readiness 曲线

### 3.2.3 重要原则

V2 中 `alpha-head` 仍然不应该成为训练主干。
它是辅助模块，不是替代 reward 的主信号。

## 3.3 V3: Dedicated Handoff Critic

### 3.3.1 动机

如果 V1/V2 发现：

- handoff reward 有效
- 但现有 value head 无法稳定估计 long-horizon handoff return

再考虑将现有 value head 从“普通 PPO critic”细化为“更显式的 handoff critic”。

### 3.3.2 V3 可能方向

- 保留原 value head，再加一个 handoff value head
- 或者保持单 head，但训练 target 明确偏向 handoff-aware return

注意：这一步不是第一版必需。

---

## 4. 环境设计

### 4.1 为什么环境是主改动点

在 `RLinf` 的 embodied 代码里，reward 是环境产出的，`task_descriptions` 也是环境产出的。
因此你这个课题的主战场其实在 env，不在 actor。

当前 `maniskill_env.py` 已经做了两件关键事情：

- 把底层环境语言指令映射到 `task_descriptions`
- 在 `step()` / `chunk_step()` 中计算 step reward

所以本实验的最小实现方式不是魔改 actor，而是：

- 新建一个 handoff-aware env，继承 `ManiskillEnv`
- 重写 `instruction` / `_wrap_obs()` / `_calc_step_reward()` / `step()` 周边逻辑

### 4.2 推荐新增环境类

建议新增：

```text
rlinf/envs/maniskill/handoff_maniskill_env.py
```

这个类的职责是：

1. 维护子任务队列
   `G = [G_1, G_2, ..., G_K]`

2. 在每个时刻维护：
   - `G_cur`
   - `G_next`
   - `alpha_t` 或其 schedule
   - 当前子任务完成状态

3. 输出给模型的观测结构保持与现有 `OpenVLA-OFT` 兼容：
   - `main_images`
   - `states`
   - `task_descriptions`

4. 在 `info` 中输出额外字段供 logging：
   - `task_complete`
   - `subtask_index`
   - `handoff_reach`
   - `handoff_reward`
   - `alpha`
   - `chain_success`

### 4.3 子任务状态机

每个 env instance 内部维护如下状态机：

```text
reset
  ->
初始化长任务 G 和当前索引 i = 0
  ->
G_cur = G[i]
G_next = G[i+1] if exists else none
  ->
rollout
  ->
如果当前子任务完成:
    i = i + 1
    更新 G_cur / G_next
    重置本子任务的 phase / alpha / handoff bookkeeping
  ->
如果 i == len(G):
    chain success
```

### 4.4 子任务完成判定

优先级建议如下：

1. simulator / task 自带 success signal
2. 基于 object pose / end-effector pose 的 rule-based detector
3. progress classifier
4. VLM-as-judge

第一版强烈建议优先用 1 或 2，不要第一版就把 completion detector 做成一个独立学习问题。

---

## 5. Prompt 设计

### 5.1 RLinf 中的 prompt 接入方式

当前 `OpenVLA-OFT` 会直接读取 `env_obs["task_descriptions"]`。
然后它内部还会自动包装成：

```text
In: What action should the robot take to {task_description}?
Out:
```

这意味着你的 handoff prompt 不需要改模型输入接口，只需要由 env 提供一条更丰富的 `task_descriptions` 字符串。

### 5.2 V1 推荐 prompt 模板

V1 推荐模板如下：

```text
current task: {G_cur}
next task: {G_next_or_none}
transition phase: {phase_text}
```

其中 `phase_text` 由脚本调度产生，例如：

- `focus on current task`
- `begin preparing for transition`
- `actively transitioning`
- `prioritize next task readiness`

### 5.3 为什么 V1 先用文本 phase，而不是 alpha 数值

原因有三点：

1. `OpenVLA-OFT` 当前天然擅长文本 prompt，不擅长额外数值 token 设计。
2. 文本 phase 更接近模型预训练时的语言接口。
3. 第一版可以避免先引入 `alpha-head`。

### 5.4 Prompt 长度配置

现有 `maniskill_ppo_openvlaoft.yaml` 中：

- `actor.model.max_prompt_length = 30`

对于双任务 prompt，这个长度通常偏小，建议在 handoff config 中提高到：

- `64` 作为起点
- 如果任务文本较长，可到 `96`

否则 prompt 会被截断，导致 `Current task / Next task / Transition phase` 信息丢失。

### 5.5 V1 对 prompt 的实际建议

不要把 prompt 写得像自然语言散文。
建议保持：

- 句子短
- 结构稳定
- 每个字段固定模板

因为这更利于比较实验，也更利于后续把 `phase_text` 替换成 `alpha-head` 输出。

---

## 6. Reward 设计

### 6.1 总原则

本实验的总 reward 保持：

$$
r_t = r_t^{done} + \alpha_t \lambda_h r_t^{handoff} + \mathbf{1}[\alpha_t > \alpha_{th}] \lambda_s r_t^{smooth}
$$

但在真正落地到 `RLinf + OpenVLA-OFT` 时，建议按两个层级推进。

## 6.2 第一层：可跑通版本

### 6.2.1 当前任务完成奖励

$$
r_t^{done} =
\begin{cases}
R_{success}, & \text{if current subtask completes at } t \\
0, & \text{otherwise}
\end{cases}
$$

这个奖励必须保持为主奖励。

### 6.2.2 Handoff reward

第一版 handoff reward 建议保留增量形式：

$$
r_t^{handoff} = Reach(o_t, G_{next}) - Reach(o_{t-1}, G_{next})
$$

但 `Reach` 的具体实现建议分两层：

1. `State-space Reach`
   如果 simulator 能暴露物体位姿、抓手位姿、接触状态，就先用状态空间构造 handoff reach。
   这是最稳妥的 debug 起点。

2. `Visual-embedding Reach`
   论文主结果再切到视觉 embedding 版本。

### 6.2.3 Smoothness penalty

$$
r_t^{smooth} = - \|a_t - a_{t-1}\|^2
$$

只在 transition phase 激活。

### 6.2.4 推荐缩放策略

第一版推荐：

- `R_success = 10.0`
- `lambda_h = 1.0` 起跑，后续 warmup 到 `5.0 ~ 10.0`
- `lambda_s = 0.05 ~ 0.1`
- `alpha_thresh = 0.2`

最重要的约束是：

> 不完成当前任务的 policy，不能因为 handoff reward 高而获得更高总回报。

## 6.3 第二层：视觉版 Reach

### 6.3.1 目标

将 handoff reward 从“手写状态距离”升级为“视觉可达性”。

### 6.3.2 推荐实现顺序

1. 离线收集每个 `G_next` 的专家起始状态 anchor
2. 用冻结编码器预计算 anchor embedding
3. rollout 时计算当前观测 embedding
4. 用 softmin 距离得到 `Reach`

### 6.3.3 编码器选择建议

从工程可行性上，第一版不建议直接把完整 `OpenVLA-OFT` backbone 复制到每个 env worker 里算 reward。
更稳妥的顺序是：

1. simulator state reward
2. 轻量冻结视觉编码器 reward
3. 如果前两者都验证成立，再尝试复用更强的 VLA 视觉特征

原因是 env worker 侧每步算 reward，额外复制大型 VLA backbone 会显著增加显存与系统复杂度。

### 6.3.4 Reward 在代码中的落点

reward 应该落在新 env 的：

- `_calc_step_reward()`
- 或 `step()` / `chunk_step()` 里对原始 reward 再包装

而不是走 `RewardWorker`。

---

## 7. Alpha / Transition 设计

## 7.1 V1：脚本化 alpha schedule

第一版建议不用学 `alpha`，而是用可解释的 schedule。

推荐两种实现：

1. `progress-based`
   如果 env 有 progress signal：

   $$
   \alpha_t = clip\left(\frac{progress_t - p_{start}}{1 - p_{start}}, 0, 1\right)
   $$

2. `window-based`
   如果 env 没有 progress signal，则在当前子任务完成前 `w` 步线性拉起：

   $$
   \alpha_t =
   \begin{cases}
   0, & t < b_i - w \\
   \frac{t - (b_i - w)}{w}, & b_i - w \le t \le b_i
   \end{cases}
   $$

V1 推荐优先使用 `window-based`，因为更稳定、更容易对照。

## 7.2 V2：学习式 alpha-head

在 V2 中增加：

- `alpha_head(hidden_features) -> alpha_hat`

训练时：

- 专家轨迹给出 `alpha_label`
- rollout 中可加入伪标签
- 将 `alpha_hat` 映射到 phase token

### 7.2.1 V2 的代码建议

建议在以下文件中加 `alpha_head`：

```text
rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py
```

复用现有 `ValueHead` 风格实现一个 `AlphaHead` 即可：

- 2 层 MLP
- 输出 1 维
- sigmoid

### 7.2.2 V2 的损失

V2 推荐只把它当 auxiliary loss：

$$
\mathcal{L}_{alpha} = \mathbb{E}[(\hat{\alpha}_t - \alpha_t^{label})^2]
$$

不要让 `alpha-head` 替代 reward 本身。

---

## 8. 训练方案

## 8.1 Stage 0：基线 checkpoint

由于当前 `RLinf` 的 VLA SFT worker 不支持 `OpenVLA-OFT`，因此默认使用：

- 外部已有 `OpenVLA-OFT` SFT checkpoint
- 或者 RLinf 官方示例里同分布的 OpenVLA-OFT checkpoint / LoRA

这也是本实验最现实的起点。

## 8.2 Stage 1：Baseline Reproduction

先不引入 handoff，复现一条纯 `OpenVLA-OFT + PPO` baseline：

- 复制 `examples/embodiment/config/maniskill_ppo_openvlaoft.yaml`
- 将 env 换成你的 long-horizon env，但 reward 只保留 `r_done`
- 验证训练能正常收敛

这一步的目的不是出结果，而是确认：

- 新 env 没问题
- PPO 跑通
- 长任务状态机没 bug
- prompt 长度没溢出

## 8.3 Stage 2：Handoff PPO V1

### 8.3.1 配置主线

建议新建：

```text
examples/embodiment/config/maniskill_ppo_openvlaoft_handoff.yaml
```

基于现有 `maniskill_ppo_openvlaoft.yaml`，主要改：

- env 类型切到新 handoff env
- `experiment_name`
- `actor.model.max_prompt_length`
- `algorithm.kl_beta`
- reward 相关超参数
- logging 项

### 8.3.2 推荐超参起点

- `loss_type: actor_critic`
- `adv_type: gae`
- `reward_type: action_level`
- `logprob_type: token_level`
- `kl_beta: 0.01 ~ 0.05`
- `critic_warmup_steps: 10 ~ 40`

`kl_beta` 在 stock config 中默认是 `0.0`，但本实验建议打开。
因为我们不希望 handoff reward 过快破坏原本的单任务能力。

### 8.3.3 Critic Warmup

现有 actor 已支持 `critic_warmup_steps`。
对 handoff 任务，建议不要用 `0` 起跑，建议从：

- `10`
- `20`
- `40`

三个值做网格。

## 8.4 Stage 3：V2 Alpha-Head

在 V1 成功后再做：

- 模型增头
- alpha 标签
- alpha logging
- alpha-driven phase token

## 8.5 Stage 4：V3 Dedicated Handoff Critic

只有在以下条件同时满足时才进入：

- V1 确认 handoff reward 提升了 chain success
- V2 确认 alpha-head 提升了稳定性或 sample efficiency
- 现有 value head 的学习明显成为瓶颈

---

## 9. 代码改动建议

## 9.1 V1 必改文件

### 新增文件

```text
RLinf/docs/openvlaoft_handoff_rl_experiment_plan.md
rlinf/envs/maniskill/handoff_maniskill_env.py
examples/embodiment/config/env/maniskill_handoff_template.yaml
examples/embodiment/config/maniskill_ppo_openvlaoft_handoff.yaml
```

### 修改文件

```text
rlinf/envs/__init__.py
```

修改内容：

- 注册新的 env type
- `get_env_cls()` 返回 `HandoffManiskillEnv`

## 9.2 V1 尽量不改的文件

以下文件 V1 尽量不要碰：

```text
rlinf/runners/embodied_runner.py
rlinf/workers/actor/fsdp_actor_worker.py
rlinf/workers/rollout/hf/huggingface_worker.py
rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py
```

这样做的好处是：

- 一旦出问题，排查范围主要在 env 与 config
- 论文第一版结果更容易归因到 reward/prompt，而不是框架魔改

## 9.3 V2 才改的文件

```text
rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py
rlinf/models/embodiment/modules/value_head.py
```

这里新增 `alpha-head` 即可，不需要大改主干。

---

## 10. 实验矩阵

## 10.1 主实验

### E0: 单任务 / 原始 baseline

- stock `OpenVLA-OFT + PPO`
- 单任务或单步目标
- 目的：确认基本能力

### E1: Long-horizon chain env + 仅 done reward

- 使用长任务 env
- prompt 只含当前任务
- reward 只有 `r_done`

这是最关键的 long-horizon baseline。

### E2: Long-horizon chain env + dual-task prompt + done reward

- 加双任务 prompt
- 不加 handoff reward

用于回答：

> 仅靠 prompt 里显式写出 `G_next`，是否已经有帮助？

### E3: Long-horizon chain env + dual-task prompt + handoff reward

- 这是 V1 核心实验
- 使用脚本化 alpha

### E4: E3 + KL

- `kl_beta > 0`

用于验证：

> KL 约束是否有助于保住单任务能力，同时稳定 RL post-training？

### E5: E3 + V2 alpha-head

- 验证 learned alpha 是否优于 scripted alpha

## 10.2 消融实验

1. `no_handoff_reward`
   去掉 `r_handoff`

2. `no_alpha_gating`
   直接用 `r_done + lambda_h r_handoff`

3. `absolute_reach`
   用 `Reach(o_t, G_next)` 而不是增量 reward

4. `no_smooth_penalty`
   验证动作平滑项是否必要

5. `state_reach vs visual_reach`
   验证 reward encoder 选择是否影响结论

6. `short_prompt vs structured_prompt`
   验证 prompt 工程本身的作用

## 10.3 评估指标

建议至少记录：

- `chain_success_rate`
- `per_subtask_success_rate`
- `handoff_success_rate`
- `avg_reach_at_boundary`
- `single_task_regression`
- `avg_episode_len`
- `return`
- `alpha_curve`（V2）

其中最重要的是：

- 完整任务链成功率
- 子任务边界处的 handoff 成功率
- 单任务能力是否退化

---

## 11. 日志与可视化建议

建议在 env 的 `info["episode"]` 中记录：

- `chain_success`
- `subtask_success_count`
- `handoff_success_count`
- `avg_handoff_reach`
- `avg_handoff_reward`
- `alpha_mean`
- `alpha_last`

另外建议保存：

- 成功链 rollout 视频
- handoff 失败视频
- 成功边界帧与失败边界帧对照

这比只看标量指标更容易判断：

- policy 是真的学会了“为下一任务做准备”
- 还是仅仅学会了某种 reward hack

---

## 12. 资源与执行建议

## 12.1 Debug 阶段

先不要直接用 stock config 的大规模设置。
建议先做一个 debug config：

- `num_nodes = 1`
- `env.train.total_num_envs = 8 ~ 16`
- `env.eval.total_num_envs = 4 ~ 8`
- `max_episode_steps` 缩短
- `rollout_epoch = 1`
- `save_interval` 较小

先确保：

- env 状态机正确
- 子任务切换正确
- prompt 没被截断
- reward 数值量级正常
- PPO loss 不爆

## 12.2 正式实验阶段

如果走近似官方 OpenVLA-OFT 配方，可参考：

- actor 占多卡
- env / rollout 分卡
- 打开 LoRA

但不建议在实验第一周就追求吞吐。
先要保证 handoff 机制本身是正确的。

---

## 13. 风险与规避

### 风险 1：长任务 env 本身不稳定

如果长任务 env 状态机设计不稳定，那么所有 reward 实验都不可信。

规避：

- 先做 `r_done only` baseline
- 先做 rule-based completion detector
- 先做少量任务链

### 风险 2：prompt 过长被截断

规避：

- 提高 `max_prompt_length`
- 控制字段数量与句长

### 风险 3：handoff reward 量级过大，毁掉基础能力

规避：

- `lambda_h` warmup
- 打开 `kl_beta`
- 保留较大的 `R_success`

### 风险 4：视觉版 Reach 计算太重

规避：

- 先做 state-space 版本
- 论文主实验再切视觉版

### 风险 5：结果无法归因

规避：

- V1 不改模型结构
- 所有新增复杂模块都后置到 V2/V3

---

## 14. 建议的落地顺序

### 第 1 周

1. 复制并跑通 `maniskill_ppo_openvlaoft` baseline
2. 新建 `handoff_maniskill_env.py`
3. 让 env 能输出双任务 prompt
4. 只用 `r_done` 跑通 chain env baseline

### 第 2 周

1. 加入 scripted alpha
2. 加入 handoff reward
3. 跑 E1 / E2 / E3
4. 调通 reward scale 与 `kl_beta`

### 第 3 周

1. 做主要消融
2. 补 state-reach / visual-reach 对照
3. 整理 boundary case 视频

### 第 4 周以后

1. 再决定是否上 alpha-head
2. 再决定是否上 dedicated handoff critic

---

## 15. 结论性建议

对当前 `RLinf` 代码现状而言，最合理的主线不是：

- 从零重写轻量 RL 框架
- 也不是第一版就实现完整 Reinformer 替代架构

而是：

1. 使用外部 `OpenVLA-OFT` SFT checkpoint
2. 直接复用 `RLinf` 的 `OpenVLA-OFT + PPO + value head + LoRA`
3. 把新增研究工作集中在 env、prompt、reward、transition 机制
4. 用 `V1 -> V2 -> V3` 的方式逐步加复杂度

一句话概括：

> 第一版实验要证明的是 handoff-aware RL post-training 的价值，而不是证明你能重新造一个 RL 框架。

---

## 16. 实施级环境伪代码

下面给出一个更接近实际编码的 `HandoffManiskillEnv` 伪代码。
这个版本故意只覆盖 V1 所需逻辑，不把 alpha-head、视觉版 reward、离线 buffer 等复杂度一次性塞进去。

```python
class HandoffManiskillEnv(ManiskillEnv):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        self.handoff_cfg = cfg.handoff
        self.chain_specs = self._load_chain_specs(cfg.handoff.chain_spec_path)
        self.curr_subtask_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.curr_chains = [None for _ in range(self.num_envs)]
        self.prev_reach = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_action = None
        self.alpha = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.chain_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.handoff_cfg.reach_mode == "state":
            self.anchor_lib = self._load_state_anchor_library(cfg.handoff.anchor_path)
        elif self.handoff_cfg.reach_mode == "visual":
            self.anchor_lib = self._load_visual_anchor_library(cfg.handoff.anchor_path)
            self.reward_encoder = self._build_frozen_reward_encoder(cfg.handoff.encoder_name)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._sample_new_chains()
        self.curr_subtask_idx[:] = 0
        self.chain_success[:] = False
        self.alpha[:] = 0.0
        self.prev_action = None
        self.prev_reach = self._compute_reach_batch(obs, self._get_next_tasks())
        obs["task_descriptions"] = self._build_prompt_batch(obs)
        return obs, info

    def _get_cur_tasks(self):
        return [chain[idx] for chain, idx in zip(self.curr_chains, self.curr_subtask_idx.tolist(), strict=False)]

    def _get_next_tasks(self):
        next_tasks = []
        for chain, idx in zip(self.curr_chains, self.curr_subtask_idx.tolist(), strict=False):
            next_tasks.append(chain[idx + 1] if idx + 1 < len(chain) else "none")
        return next_tasks

    def _compute_alpha(self, infos):
        if self.handoff_cfg.alpha_mode == "scripted_window":
            return self._window_alpha(infos)
        if self.handoff_cfg.alpha_mode == "progress":
            return self._progress_alpha(infos)
        raise NotImplementedError

    def _phase_text(self, alpha):
        if alpha < 0.2:
            return "focus on current task"
        if alpha < 0.5:
            return "begin preparing for transition"
        if alpha < 0.8:
            return "actively transitioning"
        return "prioritize next task readiness"

    def _build_prompt_batch(self, obs):
        prompts = []
        cur_tasks = self._get_cur_tasks()
        next_tasks = self._get_next_tasks()
        for i in range(self.num_envs):
            phase_text = self._phase_text(float(self.alpha[i].item()))
            prompts.append(
                f"current task: {cur_tasks[i]}\n"
                f"next task: {next_tasks[i]}\n"
                f"transition phase: {phase_text}"
            )
        return prompts

    def _compute_reach_batch(self, obs, next_tasks):
        if self.handoff_cfg.reach_mode == "state":
            return self._compute_state_reach(obs, next_tasks)
        if self.handoff_cfg.reach_mode == "visual":
            return self._compute_visual_reach(obs, next_tasks)
        raise NotImplementedError

    def step(self, actions=None, auto_reset=True):
        obs, _reward, terminations, truncations, infos = super().step(actions, auto_reset=False)

        self.alpha = self._compute_alpha(infos)
        reach_now = self._compute_reach_batch(obs, self._get_next_tasks())
        handoff_reward = reach_now - self.prev_reach
        done_reward = self._compute_done_reward(infos)
        smooth_reward = self._compute_smooth_penalty(actions)
        reward = (
            done_reward
            + self.alpha * self.handoff_cfg.lambda_h * handoff_reward
            + (self.alpha > self.handoff_cfg.alpha_thresh).float()
            * self.handoff_cfg.lambda_s
            * smooth_reward
        )

        task_complete = self._detect_task_complete(infos)
        self._advance_subtasks(task_complete, obs)

        self.prev_reach = self._compute_reach_batch(obs, self._get_next_tasks())
        self.prev_action = actions.clone() if torch.is_tensor(actions) else actions

        obs["task_descriptions"] = self._build_prompt_batch(obs)
        infos["handoff_reward"] = handoff_reward
        infos["handoff_reach"] = reach_now
        infos["alpha"] = self.alpha.clone()
        infos["task_complete"] = task_complete
        infos["chain_success"] = self.chain_success.clone()

        if torch.logical_or(terminations, truncations).any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(
                torch.logical_or(terminations, truncations), obs, infos
            )

        return obs, reward, terminations, truncations, infos
```

### 16.1 伪代码背后的最小原则

1. `task_descriptions` 始终由 env 负责构造。
2. `alpha` 在 V1 完全是 env 内部逻辑，不依赖模型输出。
3. `reward` 完全在 env 内计算。
4. actor / rollout 不需要知道“handoff 的内部定义”。

这正是用 `RLinf` 而不是重写框架的意义。

---

## 17. 配置草案

下面给出建议新增的两份配置草案。

## 17.1 环境配置草案

建议新建：

```text
examples/embodiment/config/env/maniskill_handoff_template.yaml
```

建议结构如下：

```yaml
env_type: maniskill_handoff

total_num_envs: null
auto_reset: True
ignore_terminations: False
use_rel_reward: False
use_full_state: False
seed: 0
group_size: 1
use_fixed_reset_state_ids: False
max_steps_per_rollout_epoch: 160
max_episode_steps: 160

video_cfg:
  save_video: False
  info_on_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

init_params:
  id: "YourLongHorizonManiSkillTask-v0"
  obs_mode: "rgb"
  num_envs: null

handoff:
  chain_spec_path: "/path/to/task_chains.json"
  anchor_path: "/path/to/handoff_anchor_lib.pt"
  reach_mode: "state"         # state | visual
  encoder_name: null          # 仅 visual 模式时使用
  alpha_mode: "scripted_window"  # scripted_window | progress
  alpha_window: 12
  alpha_thresh: 0.2
  lambda_h: 1.0
  lambda_s: 0.05
  success_reward: 10.0
  phase_bins: [0.2, 0.5, 0.8]
  log_boundary_frames: True
```

### 17.1.1 说明

- `env_type` 不建议复用原始 `maniskill`，而应明确注册新类型，避免把 handoff 逻辑污染通用 env。
- `reach_mode` 第一版建议从 `state` 开始。
- `alpha_window` 直接控制 scripted alpha 的线性拉起区间。

## 17.2 实验配置草案

建议新建：

```text
examples/embodiment/config/maniskill_ppo_openvlaoft_handoff.yaml
```

建议从 stock `maniskill_ppo_openvlaoft.yaml` 复制后，改成如下风格：

```yaml
defaults:
  - env/maniskill_handoff_template@env.train
  - env/maniskill_handoff_template@env.eval
  - model/openvla_oft@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 0-3
    env: 0-1
    rollout: 2-3

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "maniskill_ppo_openvlaoft_handoff"
    logger_backends: ["tensorboard"]
  max_epochs: 400
  max_steps: -1
  only_eval: False
  val_check_interval: 10
  save_interval: 20
  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  kl_beta: 0.02
  group_size: 1
  rollout_epoch: 1
  eval_rollout_epoch: 1
  reward_type: action_level
  logprob_type: token_level
  entropy_type: token_level
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"
  bootstrap_type: always
  entropy_bonus: 0.0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0
  gamma: 0.99
  gae_lambda: 0.95
  sampling_params:
    do_sample: True
    temperature_train: 1.0
    temperature_eval: 0.6
    top_k: 0
    top_p: 1.0
    repetition_penalty: 1.0
  length_params:
    max_new_token: null
    max_length: 1024
    min_length: 1

env:
  group_name: "EnvGroup"
  train:
    total_num_envs: 16
    max_episode_steps: 160
    max_steps_per_rollout_epoch: 160
  eval:
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 160
    max_steps_per_rollout_epoch: 160
    group_size: 1

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: "/path/to/openvla-oft-sft-checkpoint"
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 16
  global_batch_size: 128
  seed: 1234
  enable_offload: False
  model:
    model_path: "/path/to/openvla-oft-sft-checkpoint"
    model_type: "openvla_oft"
    add_value_head: True
    is_lora: True
    lora_rank: 32
    lora_path: null
    max_prompt_length: 96
  optim:
    lr: 1.0e-4
    value_lr: 5.0e-4
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 5.0
    critic_warmup_steps: 20
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

### 17.2.1 为什么这份配置比 stock 更保守

相比 stock OpenVLA-OFT 配置，这里故意更保守：

- `env.train.total_num_envs` 更小
- `global_batch_size` 更小
- `max_prompt_length` 更大
- `kl_beta` 非零
- `critic_warmup_steps` 非零

因为第一阶段的目标是验证机制，而不是堆吞吐。

### 17.2.2 一条非常重要的限制

当前 `RLinf` 中：

- `OpenVLA-OFT` 的 VLA SFT 不支持
- PPO 中的 `SFT co-train` 数据混合路径也只明确支持 `OpenPI`

因此本方案默认：

- `OpenVLA-OFT` 的 SFT checkpoint 来自外部
- PPO 阶段先不依赖 `sft_loss_weight` 这条 co-train 路径

如果后续需要 OpenVLA-OFT 的 RL+SFT 混合训练，需要额外扩展 actor worker 的 SFT loader。

---

## 18. Alpha-Head 的工程注意事项

这是整个方案里最容易被忽略、但最值得提前写清楚的一点。

## 18.1 循环依赖问题

你原始构想中希望：

1. `alpha-head` 预测 `alpha_t`
2. 再把 `alpha_t` 写进 prompt 的 `transition phase`
3. 然后模型根据这个 prompt 输出动作

在 `RLinf + OpenVLA-OFT` 的当前实现里，这会带来循环依赖：

- 模型的输入 prompt 需要先知道 `alpha_t`
- 但 `alpha_t` 又是模型 forward 才能得到

如果硬做，就需要同一步执行两次 forward：

1. 不带 phase token 的 forward 预测 `alpha`
2. 带 phase token 的 forward 预测动作

这会显著增加系统复杂度和推理成本。

## 18.2 推荐解法

### 方案 A：V1 主线

使用 scripted alpha。
优点：

- 没有循环依赖
- 最稳定
- 最适合做第一版论文验证

### 方案 B：V2-Lite

使用 `alpha_{t-1}` 来构造第 `t` 步 prompt。
流程如下：

1. 第 `t-1` 步 forward 输出 `alpha_hat_{t-1}`
2. 将 `alpha_hat_{t-1}` 离散成 phase text
3. 第 `t` 步 env 把这个 phase text 写进 `task_descriptions`

优点：

- 不需要双 forward
- 能把 learned alpha 真正反馈到 prompt

缺点：

- 有一步滞后

### 方案 C：独立 alpha 预测器

在 env worker 内放一个轻量 alpha predictor：

- 输入：状态 / progress / boundary features
- 输出：alpha

这时 prompt 仍然由 env 构造，policy 只负责动作。

优点：

- 工程最干净

缺点：

- alpha 不再是“共享 VLA hidden state”的头

## 18.3 文档结论

本实验建议：

- V1 用方案 A
- V2 优先尝试方案 B
- 只有明确需要时才考虑方案 C

---

## 19. PPO 数据流中的两个隐藏细节

## 19.1 `task_descriptions` 会被 rollout worker 从 `obs` 中剥离

在当前 `huggingface_worker.py` 中，rollout 会在写 trajectory 前把 `task_descriptions` 从 `obs` / `final_obs` 中 `pop` 掉。

这件事对 V1 PPO 主线不是阻塞问题，原因是：

- actor 训练主要依赖 rollout 保存下来的 `forward_inputs`
- 这些 `forward_inputs` 已经包含 tokenized prompt

所以：

- 对 on-policy PPO，现有逻辑可以工作
- 不需要为了 V1 修改 trajectory 结构

## 19.2 但它会影响后续扩展

如果你将来要做：

- off-policy replay
- hindsight relabeling
- 基于文本 prompt 的离线分析
- alpha label 的轨迹级回放

那么最好显式保留原始 prompt 文本或 phase 标签。

也就是说：

- V1：不用改
- V2/V3：再决定是否把 raw prompt 或 alpha label 存进 trajectory

---

## 20. 成功标准与退出条件

为了避免实验无限膨胀，建议为每一阶段设定明确的“成功标准”。

## 20.1 V1 成功标准

满足以下三条中的至少两条，就可以进入 V2：

1. `chain_success_rate` 相比 `done-only baseline` 有稳定提升
2. `handoff_success_rate` 提升明显
3. 单任务能力下降控制在可接受范围内

### 建议量化口径

- `chain_success_rate` 绝对提升 `>= 5%`
- 或相对提升 `>= 10%`
- 单任务成功率下降 `< 2% ~ 3%`

## 20.2 V2 成功标准

满足以下两条中的至少一条：

1. learned alpha 比 scripted alpha 提升了最终指标
2. learned alpha 明显降低训练方差或加速收敛

如果两条都不成立，则保留 scripted alpha，直接跳过 V2。

## 20.3 V3 成功标准

只有在以下情形才值得继续：

- V1/V2 已经证明 handoff reward 的方向正确
- 但现有 value head 明显是瓶颈
- 你能从训练曲线或误差分析中明确看到 critic 学不动

否则不建议在 V3 上投入过多时间。

---

## 21. 执行命令建议

### 21.1 Debug 运行

```bash
cd RLinf
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_handoff
```

### 21.2 用 Hydra override 快速调参

```bash
cd RLinf
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name maniskill_ppo_openvlaoft_handoff \
  actor.model.max_prompt_length=96 \
  algorithm.kl_beta=0.02 \
  env.train.total_num_envs=8 \
  env.eval.total_num_envs=4 \
  runner.logger.log_path=../results/debug_handoff
```

### 21.3 第一阶段建议的运行顺序

1. `done-only baseline`
2. `dual-task prompt only`
3. `dual-task prompt + handoff reward`
4. `dual-task prompt + handoff reward + KL`
5. `V2 alpha-head`

不要一开始就跳到第 4 或第 5 步。

