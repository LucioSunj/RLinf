# EmbodiedRunner `run()` 关键调用链解释

这份笔记专门解释 `EmbodiedRunner.run()` 里下面这段代码在源码层面到底做了什么：

```python
env_handle: Handle = self.env.interact(
    input_channel=self.rollout_channel,
    output_channel=self.env_channel,
)
rollout_handle: Handle = self.rollout.generate(
    input_channel=self.env_channel,
    output_channel=self.rollout_channel,
    actor_channel=self.actor_channel,
)
self.actor.recv_rollout_trajectories(
    input_channel=self.actor_channel
).wait()
rollout_handle.wait()
```

调用位置见 [embodied_runner.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/runners/embodied_runner.py#L266)。

## 总体结构

这 4 行代码不是对 3 个普通 Python 对象做同步调用，而是对 3 个 `WorkerGroup` 发起远程调用：

- `self.env`
- `self.rollout`
- `self.actor`

所以：

- `self.env.interact(...)` 会在 env worker group 上远程执行 `EnvWorker.interact`
- `self.rollout.generate(...)` 会在 rollout worker group 上远程执行 `HuggingFaceRolloutWorker.generate`
- `self.actor.recv_rollout_trajectories(...)` 会在 actor worker group 上远程执行 `EmbodiedFSDPActor.recv_rollout_trajectories`

这些调用先返回一个远程结果句柄，只有在 `.wait()` 时才真正同步等待完成。

## 这段代码里的三个 Channel

这段调用用了 3 个 channel：

- `self.env_channel`
  env worker 把 `obs/reward/done/...` 发给 rollout worker
- `self.rollout_channel`
  rollout worker 把 action chunk 发回 env worker
- `self.actor_channel`
  rollout worker 把最终整理好的 `Trajectory` 发给 actor worker

所以消息方向是：

- `env -> rollout`: `env_channel`
- `rollout -> env`: `rollout_channel`
- `rollout -> actor`: `actor_channel`

## 第一步：`self.env.interact(...)`

调用：

```python
env_handle: Handle = self.env.interact(
    input_channel=self.rollout_channel,
    output_channel=self.env_channel,
)
```

实际执行的是 [env_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/env/env_worker.py#L527) 的 `EnvWorker.interact()`：

```python
@Worker.timer("interact")
def interact(self, input_channel: Channel, output_channel: Channel):
    env_metrics = defaultdict(list)
    for epoch in range(self.rollout_epoch):
        env_outputs = self.bootstrap_step()
        for stage_id in range(self.stage_num):
            env_output: EnvOutput = env_outputs[stage_id]
            self.send_env_batch(output_channel, env_output.to_dict())

        for _ in range(self.n_train_chunk_steps):
            for stage_id in range(self.stage_num):
                raw_chunk_actions = self.recv_chunk_actions(input_channel)
                env_output, env_info = self.env_interact_step(
                    raw_chunk_actions, stage_id
                )
                self.send_env_batch(output_channel, env_output.to_dict())
                env_outputs[stage_id] = env_output
                self.record_env_metrics(env_metrics, env_info, epoch)

        self.store_last_obs_and_intervened_info(env_outputs)
        self.finish_rollout()
    ...
    return env_metrics
```

### `interact()` 内部的执行顺序

#### 1. `bootstrap_step()`

先执行 [env_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/env/env_worker.py#L453) 的 `bootstrap_step()`。

它的作用是构造第一批 `EnvOutput`：

- 如果 `auto_reset=False`，这里会真的 `reset()` 环境，拿到初始 obs
- 如果 `auto_reset=True`，这里会直接复用上一轮保存的 `last_obs_list`

返回的是一个 `EnvOutput` 列表，每个 pipeline stage 一个。

`EnvOutput` 的结构定义在 [embodied_io_struct.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/data/embodied_io_struct.py#L20)，主要字段有：

- `obs`
- `final_obs`
- `rewards`
- `dones`
- `terminations`
- `truncations`
- `intervene_actions`
- `intervene_flags`

#### 2. `send_env_batch(output_channel, env_output.to_dict())`

初始 `EnvOutput` 产生后，env worker 立刻调用 [env_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/env/env_worker.py#L387) 的 `send_env_batch()`，把 env 输出发到 `env_channel`。

`send_env_batch()` 会：

- 先根据 env 和 rollout 的 rank 映射，把一个大 batch 按目标 rollout rank 切开
- 再用 `output_channel.put(...)` 逐个发出

所以这一步是在做：

- 把 env 的初始 obs 发给 rollout worker

#### 3. `recv_chunk_actions(input_channel)`

进入主循环后，env worker 在 [env_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/env/env_worker.py#L285) 的 `recv_chunk_actions()` 中，从 `rollout_channel` 收 rollout worker 发回来的动作：

```python
for src_rank, expected_size in src_ranks_and_sizes:
    action_i = input_channel.get(
        key=CommMapper.build_channel_key(src_rank, self._rank, extra=mode),
    )
...
chunk_action = np.concatenate(chunk_action, axis=0)
```

也就是说：

- env worker 会从所有映射到自己的 rollout rank 那里收动作分片
- 再把它们拼成当前 env batch 对应的完整 `chunk_action`

#### 4. `env_interact_step(raw_chunk_actions, stage_id)`

拿到动作之后，执行 [env_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/env/env_worker.py#L197) 的 `env_interact_step()`。

关键逻辑：

```python
chunk_actions = prepare_actions(...)
obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
    self.env_list[stage_id].chunk_step(chunk_actions)
)
...
env_output = EnvOutput(
    obs=extracted_obs,
    final_obs=...,
    rewards=chunk_rewards,
    dones=chunk_dones,
    terminations=chunk_terminations,
    truncations=chunk_truncations,
    intervene_actions=intervene_actions,
    intervene_flags=intervene_flags,
)
```

这里做了 3 件关键事：

- `prepare_actions(...)`
  把模型输出动作整理成具体环境能吃的动作格式
- `self.env_list[stage_id].chunk_step(chunk_actions)`
  真正推进环境一段 action chunk
- 把结果重新包装成统一的 `EnvOutput`

这里的 `chunk_step(...)` 才是和具体环境实现真正发生交互的地方。

#### 5. 再次 `send_env_batch(...)`

新的 `EnvOutput` 产生后，env worker 再次把它发送到 `env_channel`，供 rollout worker 继续消费。

所以 `interact()` 的总体角色就是：

- 先发初始 observation
- 再不断从 `rollout_channel` 收动作
- 推进环境
- 把新的 `obs/reward/done/...` 发到 `env_channel`

## 第二步：`self.rollout.generate(...)`

调用：

```python
rollout_handle: Handle = self.rollout.generate(
    input_channel=self.env_channel,
    output_channel=self.rollout_channel,
    actor_channel=self.actor_channel,
)
```

实际执行的是 [huggingface_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L408) 的 `generate()`：

```python
async def generate(self, input_channel: Channel, output_channel: Channel, actor_channel: Channel):
    self.rollout_results = [
        EmbodiedRolloutResult(...)
        for _ in range(self.num_pipeline_stages)
    ]

    for _ in range(self.rollout_epoch):
        await self.generate_one_epoch(input_channel, output_channel)

    for stage_id in range(self.num_pipeline_stages):
        await self.send_rollout_trajectories(
            self.rollout_results[stage_id], actor_channel
        )
```

它可以拆成两段：

- 先通过 `generate_one_epoch()` 持续和 env worker 对话，积累 rollout
- 最后把累积好的 rollout 打包成 trajectory，发给 actor worker

### `generate_one_epoch()` 里真正发生了什么

核心逻辑在 [huggingface_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L315)：

```python
env_output = await self.recv_env_output(input_channel)
...
dones, rewards = self.get_dones_and_rewards(env_output)
actions, result = self.predict(env_output["obs"])
...
chunk_step_result = ChunkStepResult(...)
self.rollout_results[stage_id].append_step_result(chunk_step_result)
...
self.send_chunk_actions(output_channel, actions)
```

#### 1. `recv_env_output(input_channel)`

rollout worker 从 `env_channel` 收 env worker 发来的 env 输出。  
函数在 [huggingface_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L460)。

关键逻辑：

```python
env_output = await input_channel.get(
    key=CommMapper.build_channel_key(src_rank, self._rank, extra=mode),
    async_op=True,
).async_wait()
...
env_output = EnvOutput.merge_env_outputs(env_outputs)
```

也就是说：

- rollout worker 从多个 env rank 收到自己的输入分片
- 再合并成一个完整 batch

#### 2. `get_dones_and_rewards(env_output)`

然后执行 [huggingface_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L239) 的 `get_dones_and_rewards()`。

这个函数会：

- 从 `env_output` 里取 `dones` 和 `rewards`
- 在某些 `auto_reset + truncation` 情况下，用 value head 对 `final_obs` 做 bootstrap，修正最后一步 reward

所以 rollout worker 拿到的不只是 obs，还会顺手处理训练时实际要用的 reward/done 张量。

#### 3. `predict(env_output["obs"])`

接着执行 [huggingface_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L207) 的 `predict()`：

```python
with torch.no_grad():
    actions, result = self.hf_model.predict_action_batch(
        env_obs=env_obs,
        **kwargs,
    )
```

这里真正发生的是：

- 用当前 rollout 侧持有的 policy 模型做前向
- 输出 `actions`
- 同时输出 `result`

`result` 里通常包含：

- `prev_logprobs`
- `prev_values`
- `forward_inputs`

这些都是 actor 后续训练会用到的。

#### 4. 构造 `ChunkStepResult`

rollout worker 把这一步需要保留的信息装进 `ChunkStepResult`：

```python
chunk_step_result = ChunkStepResult(
    actions=result["forward_inputs"].get("action", None),
    dones=dones,
    rewards=rewards,
    truncations=env_output["truncations"],
    terminations=env_output["terminations"],
    prev_logprobs=result["prev_logprobs"],
    prev_values=result["prev_values"],
    forward_inputs=result["forward_inputs"],
    versions=...,
)
```

`ChunkStepResult` 定义在 [embodied_io_struct.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/data/embodied_io_struct.py#L233)。

这个对象的意义是：

- 把“这一 chunk step 的训练信息”统一保存下来

#### 5. `append_step_result(chunk_step_result)`

然后调用 [embodied_io_struct.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/data/embodied_io_struct.py#L432) 的 `EmbodiedRolloutResult.append_step_result()`：

```python
if result.actions is not None:
    self.actions.append(result.actions)
...
if result.rewards is not None:
    self.rewards.append(result.rewards)
...
if result.forward_inputs is not None:
    self.forward_inputs.append(result.forward_inputs)
```

也就是说 rollout worker 会按时间步把：

- actions
- rewards
- dones
- prev_logprobs
- prev_values
- forward_inputs

持续累积到 `EmbodiedRolloutResult` 里。

#### 6. `append_transitions(curr_obs, next_obs)`

如果启用了 transition 收集，还会额外记录 `(curr_obs, next_obs)`，见 [embodied_io_struct.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/data/embodied_io_struct.py#L490)。

#### 7. `send_chunk_actions(output_channel, actions)`

最后 rollout worker 调用 [huggingface_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L519) 的 `send_chunk_actions()`，把刚才模型预测出的动作发回 `rollout_channel`。

它内部会：

- 按目标 env rank 切 action batch
- 对每个分片调用 `output_channel.put(...)`

所以 `generate_one_epoch()` 的基本节奏是：

1. 从 env 收 `env_output`
2. 算 `dones/rewards`
3. 用 policy 算 `actions`
4. 把训练所需信息记入 `EmbodiedRolloutResult`
5. 把动作发回 env

### `generate()` 的收尾：发 trajectory 给 actor

rollout 和 env 对话完成后，`generate()` 会调用 [huggingface_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L305) 的 `send_rollout_trajectories()`：

```python
trajectories: Trajectory = rollout_result.to_splited_trajectories(
    self.actor_split_num
)
for trajectory in trajectories:
    channel.put(trajectory, async_op=True)
```

这里关键是 [embodied_io_struct.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/data/embodied_io_struct.py#L546) 的 `to_splited_trajectories()`：

- 先通过 `to_trajectory()` 把所有按时间累积的 list stack 成一个 `Trajectory`
- 再按 batch 维切成多个 shard

`Trajectory` 里包括：

- `actions`
- `rewards`
- `dones`
- `prev_logprobs`
- `prev_values`
- `forward_inputs`
- `curr_obs`
- `next_obs`

最后这些 `Trajectory` 被写入 `actor_channel`。

## 第三步：`self.actor.recv_rollout_trajectories(...).wait()`

调用：

```python
self.actor.recv_rollout_trajectories(
    input_channel=self.actor_channel
).wait()
```

实际执行的是 [fsdp_actor_worker.py](/Users/lucio/Desktop/Projects/EmbodiedWorld/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L1083) 的 `recv_rollout_trajectories()`：

```python
send_num = self._component_placement.get_world_size("rollout") * self.stage_num
recv_num = self._component_placement.get_world_size("actor")
split_num = compute_split_num(send_num, recv_num)

recv_list = []
for _ in range(split_num):
    trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
    recv_list.append(trajectory)

self.rollout_batch = convert_trajectories_to_batch(recv_list)
self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
```

这段逻辑的意思是：

- 先根据 rollout worker 数和 actor worker 数，算出当前 actor rank 应该收几个 trajectory shard
- 然后从 `actor_channel` 里逐个 `get()`
- 收齐后，用 `convert_trajectories_to_batch(...)` 合并成一个训练 batch
- 再 `_process_received_rollout_batch(...)` 做进一步整理

`_process_received_rollout_batch(...)` 会做一些 actor 训练前需要的标准化工作，比如：

- 重排时间维和 batch 维
- 计算 `loss_mask`
- 按 reward 过滤样本

所以这一步之后，actor worker 内部的 `self.rollout_batch` 才真正准备好，可用于后面的：

- `compute_advantages_and_returns()`
- `run_training()`

### 为什么这里必须 `.wait()`

因为后面 runner 紧接着就要调用：

```python
self.actor.compute_advantages_and_returns().wait()
```

如果不在这里等待接收完成，那么 actor 可能还没有把 trajectory 全部读完、拼完、整理完，后面的 advantage 计算就没有合法输入。

所以这里的 `.wait()` 不是可选的语义糖，而是一个明确的同步点：

- 保证 actor 已经收齐并处理好 rollout 结果

## 第四步：`rollout_handle.wait()`

最后还有：

```python
rollout_handle.wait()
```

这表示：

- 虽然 actor 已经把需要的 trajectory 收到手了
- 但 rollout worker 自己的 `generate()` 整体调用未必已经完全结束

比如它还可能在做：

- 所有 stage 的 trajectory 发送收尾
- offload model

所以这里再等一次，是为了保证：

- rollout 阶段整体结束
- runner 才进入下一阶段

## 把 4 行代码串成一条完整时序

### `env worker`

- `bootstrap_step()` 先产出初始 `EnvOutput`
- 通过 `env_channel` 把初始 obs 发给 rollout
- 持续从 `rollout_channel` 收动作
- 调 `chunk_step()` 推进环境
- 再通过 `env_channel` 发新的 `obs/reward/done`

### `rollout worker`

- 从 `env_channel` 收 `EnvOutput`
- 用 `predict()` 调 policy 得到动作
- 用 `EmbodiedRolloutResult` 累积训练信息
- 把动作发回 `rollout_channel`
- 最后把累计结果切成 `Trajectory` 发到 `actor_channel`

### `actor worker`

- 从 `actor_channel` 收 `Trajectory`
- 合并成 `rollout_batch`
- 做训练前整理
- 供后续 advantage 和 training 使用

## 一句话总结

`EmbodiedRunner.run()` 里的这段代码，本质上是在驱动一条三段式流水线：

1. `env worker` 负责和环境真正交互
2. `rollout worker` 负责用 policy 基于 env 输出生成动作，并积累 rollout 数据
3. `actor worker` 负责接收 rollout 最终产物，并整理成训练 batch

其中：

- `env_channel` 传 env 输出
- `rollout_channel` 传动作
- `actor_channel` 传最终 trajectory

而 `.wait()` 的作用就是在需要的时候把这条流水线同步住。
