# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle as pkl
import queue
import threading
import time
from typing import Any, Callable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import concat_batch

logger = get_logger()


def _is_normalizable_obs_tensor(tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor):
        return False
    if tensor.dtype not in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ):
        return False
    if tensor.ndim >= 3:
        return False
    return True


def compute_observation_stats(
    replay_buffer: TrajectoryReplayBuffer,
    eps: float = 1e-6,
) -> dict[str, dict[str, torch.Tensor]]:
    stats: dict[str, dict[str, torch.Tensor]] = {}
    trajectory_ids = list(getattr(replay_buffer, "_trajectory_id_list", []))

    for trajectory_id in trajectory_ids:
        flat_trajectory = None
        if replay_buffer._flat_trajectory_cache is not None:
            flat_trajectory = replay_buffer._flat_trajectory_cache.get(trajectory_id)

        if flat_trajectory is None:
            trajectory_info = replay_buffer._trajectory_index[trajectory_id]
            model_weights_id = trajectory_info["model_weights_id"]
            trajectory = replay_buffer._load_trajectory(trajectory_id, model_weights_id)
            flat_trajectory = replay_buffer._flatten_trajectory(trajectory)

        for obs_key in ("curr_obs", "next_obs"):
            obs_dict = flat_trajectory.get(obs_key, None)
            if not isinstance(obs_dict, dict):
                continue

            for key, value in obs_dict.items():
                if not _is_normalizable_obs_tensor(value):
                    continue

                flattened = value.float().reshape(value.shape[0], -1)
                if key not in stats:
                    stats[key] = {
                        "sum": flattened.sum(dim=0),
                        "sum_sq": flattened.square().sum(dim=0),
                        "count": torch.tensor(float(flattened.shape[0])),
                        "shape": torch.tensor(list(value.shape[1:]), dtype=torch.long),
                    }
                else:
                    stats[key]["sum"] += flattened.sum(dim=0)
                    stats[key]["sum_sq"] += flattened.square().sum(dim=0)
                    stats[key]["count"] += float(flattened.shape[0])

    normalized_stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, entry in stats.items():
        count = torch.clamp(entry["count"], min=1.0)
        mean = entry["sum"] / count
        var = torch.clamp(entry["sum_sq"] / count - mean.square(), min=0.0)
        target_shape = tuple(entry["shape"].tolist())
        normalized_stats[key] = {
            "mean": mean.reshape(target_shape),
            "std": torch.sqrt(var + eps).reshape(target_shape),
        }
    return normalized_stats


def apply_observation_normalizer(
    batch: dict[str, Any],
    observation_stats: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if not observation_stats:
        return batch

    for obs_key in ("curr_obs", "next_obs"):
        obs_dict = batch.get(obs_key, None)
        if not isinstance(obs_dict, dict):
            continue

        for key, stats in observation_stats.items():
            if key not in obs_dict:
                continue
            tensor = obs_dict[key]
            if not torch.is_tensor(tensor):
                continue
            mean = stats["mean"].to(device=tensor.device, dtype=tensor.dtype)
            std = stats["std"].to(device=tensor.device, dtype=tensor.dtype)
            obs_dict[key] = (tensor - mean) / std
    return batch


def _load_offline_trajectory_file(path: str) -> list[Trajectory]:
    if path.endswith((".pt", ".pth")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    elif path.endswith(".pkl"):
        with open(path, "rb") as file_obj:
            payload = pkl.load(file_obj)
    else:
        raise ValueError(
            f"Unsupported offline trajectory file format for '{path}'. "
            "Supported formats: .pt, .pth, .pkl"
        )

    if isinstance(payload, Trajectory):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, Trajectory) for item in payload):
            raise TypeError("Offline trajectory list must contain Trajectory objects.")
        return payload
    if isinstance(payload, dict):
        for key in ("episodes", "trajectories", "data"):
            value = payload.get(key, None)
            if isinstance(value, list) and all(
                isinstance(item, Trajectory) for item in value
            ):
                return value
    raise TypeError(
        "Offline trajectory payload must be a Trajectory, a list[Trajectory], "
        "or a dict containing an 'episodes'/'trajectories' list."
    )


def _load_offline_trajectory_payload(path: str) -> list[Trajectory]:
    if os.path.isdir(path):
        trajectories: list[Trajectory] = []
        supported_suffixes = (".pt", ".pth", ".pkl")
        trajectory_files = []
        for root, _, file_names in os.walk(path):
            for file_name in sorted(file_names):
                if file_name.endswith(supported_suffixes):
                    trajectory_files.append(os.path.join(root, file_name))

        if not trajectory_files:
            raise ValueError(
                f"No offline trajectory files were found in directory '{path}'. "
                "Supported formats: .pt, .pth, .pkl"
            )

        for trajectory_file in trajectory_files:
            trajectories.extend(_load_offline_trajectory_file(trajectory_file))
        return trajectories

    return _load_offline_trajectory_file(path)


class ReplayBufferDataset(IterableDataset):
    """Dataset that samples batches from replay and demonstration buffers.

    This dataset provides an infinite iterator that yields batches sampled from
    a replay buffer and optionally a demonstration buffer. When both buffers are
    provided, batches are composed of half replay samples and half demonstration
    samples.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing offline demonstration trajectories
            and online human-in-the-loop trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        batch_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the ReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch. When demo_buffer is
                provided, batch_size // 2 samples come from each buffer.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            **kwargs: Additional keyword arguments (unused, for compatibility).
        """
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size
        self.batch_transform = batch_transform

    @classmethod
    def from_offline_path(
        cls,
        path: str,
        *,
        batch_size: int,
        min_replay_buffer_size: int = 0,
        min_demo_buffer_size: int = 0,
        prefetch_size: int = 5,
        seed: int = 1234,
        enable_cache: bool = True,
        cache_size: int = 5,
        sample_window_size: int = 0,
        trajectory_format: str = "pt",
        state_normalizer: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        batch_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> "ReplayBufferDataset":
        metadata_path = os.path.join(path, "metadata.json")
        index_path = os.path.join(path, "trajectory_index.json")
        use_checkpoint = (
            os.path.isdir(path)
            and os.path.exists(metadata_path)
            and os.path.exists(index_path)
        )
        trajectories = None
        effective_enable_cache = bool(enable_cache)
        if not use_checkpoint:
            trajectories = _load_offline_trajectory_payload(path)
            cache_size = max(int(cache_size), len(trajectories))
            # Raw trajectory files do not carry replay-buffer checkpoint metadata, so
            # samples must remain cache-backed after the initial load.
            effective_enable_cache = True

        replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=effective_enable_cache,
            cache_size=cache_size,
            sample_window_size=sample_window_size,
            auto_save=False,
            trajectory_format=trajectory_format,
        )

        if use_checkpoint:
            replay_buffer.load_checkpoint(path)
        else:
            replay_buffer.add_trajectories(trajectories)

        def _offline_batch_transform(batch: dict[str, Any]) -> dict[str, Any]:
            if reward_scale != 1.0 or reward_bias != 0.0:
                batch["rewards"] = batch["rewards"] * reward_scale + reward_bias
            if state_normalizer is not None:
                batch = apply_observation_normalizer(batch, state_normalizer)
            if batch_transform is not None:
                batch = batch_transform(batch)
            return batch

        return cls(
            replay_buffer=replay_buffer,
            demo_buffer=None,
            batch_size=batch_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=prefetch_size,
            batch_transform=_offline_batch_transform,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an infinite iterator that yields batches.

        Waits until both buffers (if demo_buffer is provided) reach their
        minimum size requirements before yielding batches. When ready, samples
        from replay buffer only or from both replay and demo buffers.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        while True:
            is_ready = True
            if not self.replay_buffer.is_ready(self.min_replay_buffer_size):
                is_ready = False
            if self.demo_buffer is not None and not self.demo_buffer.is_ready(
                self.min_demo_buffer_size
            ):
                is_ready = False

            if is_ready:
                if self.demo_buffer is not None:
                    replay_batch = self.replay_buffer.sample(self.batch_size // 2)
                    demo_batch = self.demo_buffer.sample(self.batch_size // 2)
                    batch = concat_batch(replay_batch, demo_batch)
                else:
                    batch = self.replay_buffer.sample(self.batch_size)
                if self.batch_transform is not None:
                    batch = self.batch_transform(batch)
                yield batch

    def close(self) -> None:
        """Releases references to replay and demo buffers."""
        del self.replay_buffer
        del self.demo_buffer

    def __del__(self) -> None:
        """Destructor that ensures buffers are cleaned up."""
        self.close()


class PreloadReplayBufferDataset(ReplayBufferDataset):
    """Dataset that prefetches batches from replay and demo buffers in background.

    This dataset extends ReplayBufferDataset by prefetching batches in a
    background thread, which can improve throughput by overlapping sampling
    with training. Batches are stored in a queue of configurable size.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing demonstration trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
        prefetch_size: Maximum number of batches to prefetch and store in queue.
        preload_queue: Queue holding prefetched batches.
        sample_thread: Background thread that samples batches.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        prefetch_size: int = 5,
        batch_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        """Initializes the PreloadReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch. When demo_buffer is
                provided, batch_size // 2 samples come from each buffer.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            prefetch_size: Maximum number of batches to prefetch and store in
                the queue. Defaults to 10.
        """
        self._stop_event = threading.Event()

        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size
        self.batch_transform = batch_transform
        self.prefetch_size = prefetch_size
        assert self.prefetch_size > 0, f"{self.prefetch_size=} must be greater than 0"

        self.preload_queue = queue.Queue(maxsize=prefetch_size)
        self.sample_thread = None
        self._exception = None

    def _sample_buffer(self) -> None:
        """Background thread target that continuously samples batches.

        Runs in a loop until stop event is set. Waits for buffers to be ready,
        samples batches, and puts them in the preload queue. If the queue is
        full, skips the sample and retries. Sleeps when buffers are not ready
        or when errors occur.
        """
        while not self._stop_event.is_set():
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            is_ready = True
            if not self.replay_buffer.is_ready(self.min_replay_buffer_size):
                is_ready = False
            if self.demo_buffer is not None and not self.demo_buffer.is_ready(
                self.min_demo_buffer_size
            ):
                is_ready = False

            if is_ready:
                if self.demo_buffer is not None:
                    replay_batch = self.replay_buffer.sample(self.batch_size // 2)
                    demo_batch = self.demo_buffer.sample(self.batch_size // 2)
                    batch = concat_batch(replay_batch, demo_batch)
                else:
                    batch = self.replay_buffer.sample(self.batch_size)
                if self.batch_transform is not None:
                    batch = self.batch_transform(batch)
            else:
                time.sleep(3)
                continue

            try:
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                logger.info("Queue is full, skipping sample")
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Error in ReplayBufferDataset: {e}")
                self._exception = e
                self._stop_event.set()
                break

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an iterator that yields prefetched batches.

        Starts the background sampling thread on first call. Retrieves batches
        from the preload queue and yields them. Stops when the stop event is set.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(
                target=self._sample_buffer, daemon=True
            )
            self.sample_thread.start()

        while not self._stop_event.is_set():
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    # Check if thread died with exception
                    if hasattr(self, "_exception"):
                        raise RuntimeError(
                            "Sampling thread failed"
                        ) from self._exception
                    break
                continue

    def close(self) -> None:
        """Stops the background sampling thread and cleans up resources.

        Sets the stop event and waits up to 10 seconds for the sampling thread
        to terminate. Logs a warning if the thread does not terminate in time.
        """
        self._stop_event.set()

        thread_timeout = 10
        if self.sample_thread is not None and self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    f"Sample thread is still alive after {thread_timeout} seconds, force killing"
                )

    def __del__(self) -> None:
        """Destructor that ensures the sampling thread is stopped."""
        if not self._stop_event.is_set():
            self.close()


def replay_buffer_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader that returns the first batch element.

    Since the dataset already yields complete batches, this function simply
    extracts the batch from the list wrapper added by DataLoader.

    Args:
        batch: List containing a single batch dictionary.

    Returns:
        The unwrapped batch dictionary.
    """
    return batch[0]
