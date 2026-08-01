# Causal Shift Audit

- manual shifted CE: 0.02505086548626423
- model returned loss: 0.02505086362361908
- abs diff: 1.862645149230957e-09
- shifted answer token accuracy: 0.9979661107063293
- shifted first answer token accuracy: 1.0
- shifted remaining answer token accuracy: 0.9982674982674983
- shifted EOS accuracy: 0.96875
- sequence-level teacher-forced exact match: 0.90625

The audit uses `shift_logits = logits[:, :-1, :]` and `shift_labels = labels[:, 1:]`.
