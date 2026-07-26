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

"""Frozen-weight steering interventions on the UNCOND sampling ODE (WS6).

Nothing here touches model weights; every intervention acts at inference time
on the frozen action expert's solver loop.  Consumers import via the full path
(``rlinf.models.embodiment.wam_policy.steering``) — the parent ``wam_policy``
``__init__`` deliberately does not re-export this package.
"""

from .flow_guidance import (
    GuidanceConfig,
    LinearProbeCritic,
    constant_lambda_after_threshold,
    make_q_guidance_hook,
    predicted_clean_action,
)

__all__ = [
    "GuidanceConfig",
    "LinearProbeCritic",
    "constant_lambda_after_threshold",
    "make_q_guidance_hook",
    "predicted_clean_action",
]
