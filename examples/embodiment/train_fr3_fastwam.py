# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run task-bound mixed FR3 RL after assets, limits and bridge are configured."""

import hydra
from omegaconf import DictConfig

from rlinf.models.embodiment.wam_policy.fr3_runner import Fr3SynchronousRunner


@hydra.main(
    version_base="1.3", config_path="config", config_name="realworld_fr3_fastwam_drawer"
)
def main(cfg: DictConfig) -> None:
    Fr3SynchronousRunner(cfg).run()


if __name__ == "__main__":
    main()
