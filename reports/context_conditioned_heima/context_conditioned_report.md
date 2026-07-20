# Context-Conditioned Heima Baseline

Strict core stayed locked: predictor hidden, official projector, official embedding replacement, torchtune chunked CE.

Semantic gate allow Ours-L1 rerun: True

This run restores Heima-style information asymmetry: Model A sees context+question; Model B sees question+latent only.

Supplemental full semantic gate: free-generation normal did not outperform paired shuffle, so the requested full gate remains false despite NLL-only gate passing.
