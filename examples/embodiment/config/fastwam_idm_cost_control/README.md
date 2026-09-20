# FastWAM IDM cost-control profiles

## Release cost inside the target band

`band_price_in_band_decay_b50` adds in-band cost release to the existing
reversal-damped controller. For route-neutral training with critic warm-up,
select `pad_route_neutral_warmup_in_band_decay_b50` instead. Both keep the
target `0.5`, half-width `0.03`, and expected eligible behavior-probability
feedback. The interval includes both edges: `[0.47, 0.53]`.

After each post-warm-up rollout whose feedback EMA lies inside the band:

```text
next_price = in_band_decay_factor * signed_price
next_idm_cost = max(next_price, 0)
next_uncond_cost = max(-next_price, 0)
```

The new profiles set `signed_price.in_band_decay_factor: 0.0`, so both next
costs become exactly zero. Set it to `0.2` to retain 20% on every in-band
rollout (4% after two). An omitted factor or `1.0` retains historical behavior.
The supported range is `[0, 1]`. Feedback selection and EMA semantics are
unchanged; the profiles inherit `ema_beta: 0.0` and thus use the latest
rollout's expected eligible IDM fraction.

Release bypasses `update_interval` and `max_delta_per_update`, and restarts
the interval counter. This ensures even a large accumulated price can clear
on entry. Outside the band, the original bounded feedback and opposite-side
reversal decay resume. Warm-up still freezes controller history and publishes
zero costs. An observation from rollout `t` affects costs starting at rollout
`t + 1`; it never changes rewards for the completed rollout.

For `libero_10_ppo_fastwam_route_neutral_online_formal` or
`libero_10_ppo_fastwam_pad_route_neutral_formal`, replace the existing group:

```text
fastwam_idm_cost_control=pad_route_neutral_warmup_in_band_decay_b50
```

For fast decay, add:

```text
algorithm.fixed_branch_cost.controller.signed_price.in_band_decay_factor=0.2
```

For the generic adaptive configuration, add the group with
`+fastwam_idm_cost_control=band_price_in_band_decay_b50`. Target and half-width
remain configurable under `controller.rate`. Existing performance overlays
retain their own reversal factor and scientific settings.

These profiles retain their parent controller types; the resolved config is
already bound into checkpoints. Same-profile resume is supported; changing
this factor is not an exact continuation of an older checkpoint. Audit JSONL
and metrics expose `in_band_decay_applied`, `in_band_decay_factor`, and
`in_band_decay_delta`. The delta describes release for the next rollout;
the usual cost metrics describe the cost applied to the completed rollout.
This releases accumulated price pressure; reduced oscillation in training
remains an experimental hypothesis, since policy updates also have dynamics.

## Release cost after an opposite-side crossing

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
