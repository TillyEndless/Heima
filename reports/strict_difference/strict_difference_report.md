# STRICT-HEIMA-DIFFERENCE-EXPERIMENT

Status: complete. Loss2 was not run.

Strict repo remained locked to `thinking_state_mode=predictor`, `<THINKING_OF_REASONING>`, `HeimaOfficialAbstractProjection`, and official Torchtune chunked CE with `fallback_used=false`.

S0 and S1 completed for seeds [42, 43, 44]. J-detach and J-no-detach were forked from each seed's same S1 checkpoint and their structured configs differed only by `detach_encoder_latent`.

Loss1-to-A check:
- detach branch grad_A_from_loss1: [0.0, 0.0, 0.0]
- no-detach branch grad_A_from_loss1: [0.03598966314641634, 0.056855926729408604, 0.05628842393965754]

Cross-seed paired summary is in `cross_seed_summary.json`; raw per-seed metrics are in `seed_results.json`.
