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

"""Hydra entrypoint for closed-loop FastWAM routing-budget calibration."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from rlinf.runners.fastwam_budget_calibration import (
    run_fastwam_routing_budget_suite,
)


@hydra.main(
    version_base=None,
    config_path="../examples/embodiment/config",
    config_name="libero_10_ppo_fastwam_adaptive",
)
def main(cfg: DictConfig) -> None:
    """Run the config-selected calibration suite through the existing evaluator."""

    run_fastwam_routing_budget_suite(
        cfg,
        repo_root=Path(__file__).resolve().parents[1],
    )


if __name__ == "__main__":
    main()
