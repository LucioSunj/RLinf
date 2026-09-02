# FastWAM IDM cost-control profiles

`band_price_reversal_damped_b50` is an explicit target-rate ablation derived
from `band_price_b50`. It keeps the B50 target, `0.03` half-width, expected
eligible-rate feedback, one-rollout lag, and mutually exclusive non-negative
IDM/UNCOND branch costs.

When the current signed price and the newly observed band error have opposite
signs, the profile uses half of the historical price as the next update's base:

```text
base_price = 0.5 * signed_price  if band_error * signed_price < 0
base_price = signed_price        otherwise
next_price = project(base_price + learning_rate * band_error)
```

The total price change, including the decay, remains subject to
`max_delta_per_update`, followed by the existing signed-price projection. The
rate observed after rollout `t` can therefore affect costs only on rollout
`t + 1`.

Select it through Hydra without editing Python:

```text
+fastwam_idm_cost_control=band_price_reversal_damped_b50
```

This profile has a distinct controller type and checkpoint identity. It is not
an exact-resume replacement for `band_price_b50`, and its gain and decay factor
remain experimental rather than calibrated formal defaults.
