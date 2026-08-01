# Checkpoint Retention Plan

No checkpoints were deleted.

Recommended O0 retention: keep `final_adapter`, one early checkpoint `step50`, and one mid checkpoint `step250`. Other O0 step checkpoints are candidates for deletion after user approval.

| path | size | role | recommendation |
|---|---:|---|---|
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/final_adapter` | 6.2G | O0 checkpoint | keep |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step50` | 6.2G | O0 checkpoint | keep |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step250` | 6.2G | O0 checkpoint | keep |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step100` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step150` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step200` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step300` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step350` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step400` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step450` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_o0/step500` | 6.2G | O0 checkpoint | candidate delete after user approval |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_v2_direct/best_adapter` | 6.2G | direct checkpoint | keep |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_repair_v2_direct/final_adapter` | 6.2G | direct checkpoint | keep |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/direct_generation_closure_continued/best_adapter` | 6.2G | direct checkpoint | keep |
| `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/direct_generation_closure_continued/final_adapter` | 6.2G | direct checkpoint | keep |

Approximate releasable space from O0 candidate step checkpoints: about 8 * 6.2G = 49.6G, subject to user confirmation.
