# Direct Generation Closure Report

## Decision

CASE 3: Direct gate passed, but repaired O0 Forced-K failed. Next round may run O1 balanced / O2 answer-heavy. Do not start 10k pilot or Stage2 yet.

## Direct Gate

- Existing direct final: 29/32 at step20 after continuation context; original existing final was 29/32.
- Continuation used direct-answer NTP only, lr=5e-06, stopped at 40 additional steps.
- Final token exact: 31/32 = 0.96875
- Final normalized exact: 31/32 = 0.96875
- Valid generation rate: 1.0
- Clean reload reproducible: True

## Original Three Direct Failures

| sample | class | type | divergence | gold token | generated token | interpretation |
|---|---|---|---:|---|---|---|
| 67 | interior_token_error | natural_language | 70 | `店内` | `餐厅` | genuine local token choice error |
| 181 | interior_token_error | natural_language | 104 | ` 

` | ` Kad` | genuine local token choice error |
| 237 | missing_eos | natural_language | 242 | `<｜end▁of▁sentence｜>` | ` and` | content matched until EOS region, failure is EOS/stop token |

## Cache Divergence

- Divergent cache/no-cache cases: 2
- Divergence occurs even in single-sample generation, so this is not merely batched padding. Validity evaluation should continue to use `use_cache=False`.

## Repaired O0 Evaluation

- O0 shifted teacher-forced accuracy: 0.9921086709946394
- Forced-K token exact: 6/32 = 0.1875
- Forced-K type-aware accuracy: 8/32 = 0.25
- Fixed-K token exact: 0/32 = 0.0
- Free-K mean generated THINK count: 225.90625
- Free-K stop success rate: 0.25

## Required Answers

1. direct 剩余 3 条为什么失败？两条 interior token error，一条 missing EOS；见上表。
2. 它们是真错误还是格式等价？原 direct final 的三条不是简单 whitespace/parser 问题；继续 direct 40 step 后减少到 1/32。
3. cache divergence 根因是什么？单样本 cache/no-cache 也有 2 条 divergence；不是 batched padding alone。正式 validity 用 no-cache。
4. direct gate 是否通过？通过，31/32 token exact、31/32 normalized exact、valid generation 1.0、reload 可复现。
5. 修复后的 O0 Forced-K 是否通过？不通过，token exact 0.1875。
6. 下一步是 O1/O2、boundary 实验，还是恢复小规模 M0？下一步可以跑 O1/O2；不恢复小规模 M0，因为 O0 Forced-K 未过。
7. 是否仍然禁止 10k pilot 和 Stage2？是。
