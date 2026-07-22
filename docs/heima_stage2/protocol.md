# Experiment Protocol

## Matrix

| experiment | Stage1 | Stage2 mode | A | B | total loss |
|---|---|---|---|---|---|
| A | train B with frozen A | `heima_baseline` | trainable | frozen B* | `L_NTP` |
| B | same B* | `ours_interp_supervision` | trainable | frozen B* | `L_NTP + lambda_interp * L_interp` |

Both experiments must use the same A initialization, B* checkpoint, dataset,
optimizer hyperparameters, schedule, training steps, validation split, and
generation parameters. The only allowed algorithmic difference is whether the
latent `z` passed into frozen B is detached.

## Stop Gates

- B has any trainable parameter in Stage2.
- B parameters appear in the optimizer.
- `grad_A_from_interp != 0` for `heima_baseline`.
- `grad_A_from_interp <= 0` or non-finite for `ours_interp_supervision`.
- The two Stage2 modes use different data, checkpoints, optimizer settings, or
  training steps.

## Outputs

- `stage1_manifest.json`
- `stage2_heima_baseline_manifest.json`
- `stage2_ours_interp_supervision_manifest.json`
- gradient audit JSON
- answer accuracy / NTP loss / interpreter loss
- interpreter generation metrics and correct/shuffle/zero intervention metrics
