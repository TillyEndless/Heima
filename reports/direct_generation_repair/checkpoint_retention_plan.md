# Checkpoint Retention Plan

No files were deleted.

| path | size | role | recommendation |
|---|---:|---|---|
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0` | 68G | O0 every-50-step plus final | Potentially delete intermediate step50..step500 after user approval; keep final_adapter if needed. |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_v2_direct` | 13G | direct-answer best/final | Keep until direct repair is resolved; only best/final exist. |
