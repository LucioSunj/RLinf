# LingBot-VA single-arm route-neutral policy

The registered model is `lingbot_va_route_neutral`. `run.py` provides collection,
data preparation, LeRobot export, native parent SFT, causal UNCOND BC, single-robot
online RL, WebSocket deployment, six-method evaluation and recorded-data validation.

See the complete experiment guide in the workspace at
[`docs/LINGBOT_VA_SINGLE_ARM.md`](../../../../docs/LINGBOT_VA_SINGLE_ARM.md).

`config.yaml` records the selected scientific settings. Fill the asset environment
variables documented in the guide. Copy `robot.yaml` and fill the existing calibrated
Franka settings on the CPU controller machine. GPU commands require explicitly idle
physical GPU IDs0–6; GPU7 is excluded from both queries and allocation.

Implementation tests use CPU tiny native models. Real-weight checks, real-robot
training and main evaluation require the user's assets and hardware and have not run.
