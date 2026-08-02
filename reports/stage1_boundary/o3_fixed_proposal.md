# O3-Fixed Proposal

Do not train automatically. Proposed fixed protocol: choose K=64 or debug32 P50 K, train and evaluate all samples with the same K, keep explicit `<THINK_END><ANSWER>` boundary, and compare against dynamic-K O3 using the same base checkpoint, seed, LoRA config, optimizer, max steps, and no-cache evaluator. Fixed-K metrics should only be formal for a separately trained O3-Fixed model.
