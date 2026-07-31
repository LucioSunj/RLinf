# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small fail-fast contracts shared by RLinf configuration entry points."""

from pathlib import Path


def validate_pi05_critic_artifact_config(
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> None:
    path = str(checkpoint_path).strip()
    digest = str(checkpoint_sha256).strip().lower()
    if not path:
        raise ValueError("The pi0.5 critic requires a non-empty checkpoint path.")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("The pi0.5 critic requires a 64-character hexadecimal SHA-256.")


def validate_fastwam_kv_weight_sync(
    backend: str, weight_sync_interval: int
) -> None:
    backend = str(backend).lower()
    interval = int(weight_sync_interval)
    if interval < 1:
        raise ValueError("`runner.weight_sync_interval` must be positive.")
    if backend == "recompute" and interval != 1:
        raise ValueError(
            "FastWAM K/V recomputation requires `runner.weight_sync_interval: 1` "
            "so the rollout actor version can be reconstructed exactly."
        )


def validate_libero_terminal_reward_config(
    *,
    ignore_terminations: bool,
    use_rel_reward: bool,
) -> None:
    if bool(ignore_terminations) and not bool(use_rel_reward):
        raise ValueError(
            "FastWAM adaptive forbids `ignore_terminations: true` with "
            "`use_rel_reward: false` because terminal success rewards would repeat."
        )


def validate_fastwam_resume_steps(
    loaded_steps,
    resume_dir: str,
) -> int:
    """Use the cross-rank checkpoint payload step and audit the directory label."""

    values = (
        list(loaded_steps)
        if isinstance(loaded_steps, (list, tuple))
        else [loaded_steps]
    )
    if not values or any(
        value is None or isinstance(value, bool) for value in values
    ):
        raise ValueError("FastWAM checkpoint workers did not return payload steps.")
    steps = {int(value) for value in values}
    if len(steps) != 1:
        raise ValueError(
            f"FastWAM checkpoint ranks disagree on payload step: {sorted(steps)}."
        )
    payload_step = steps.pop()
    directory_name = Path(str(resume_dir)).name
    prefix = "global_step_"
    if not directory_name.startswith(prefix):
        raise ValueError(
            "FastWAM resume directory must be named `global_step_<payload step>`."
        )
    try:
        directory_step = int(directory_name.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(
            "FastWAM resume directory must end in an integer payload step."
        ) from exc
    if directory_step != payload_step:
        raise ValueError(
            "FastWAM resume directory/payload step mismatch: "
            f"directory={directory_step}, payload={payload_step}."
        )
    return payload_step
