# Stage-1 Repair V2 Report

Can resume 10k pilot: **False**

## Key Findings

- O0 transition audit shows the model still predicts `<THINK>` after `Question + K*<THINK>`: `<THINK>` prob=0.9601, expected answer-boundary token prob=0.0396.
- O0 teacher-forcing is high after the boundary, but the first answer/boundary transition is poor: first-answer-token acc=0.000, NLL=3.5301; remaining answer token acc=1.000.
- Full forward vs generate logits are consistent: max prob diff use_cache=True 1.67e-06, use_cache=False 1.67e-06; top-20 lists are identical. So this is not a cache/generate implementation mismatch.
- Direct-answer control failed despite teacher-forced answer token acc=1.000: free answer acc=0.4062, valid answer rate=0.5625.

## Required Answers

1. THINK->Answer first token 是否是主要失败点？  
   是，O0 在 `Q + K THINK` 后仍把 `<THINK>` 排 rank 1，答案边界 token 只排 rank 2；first-answer/boundary transition 明显差，而 boundary 后 remaining answer token teacher-forcing 很好。

2. training/generation template 是否一致？  
   O0 旧 Forced-K/Fixed-K generation 预先提供了 `Answer:` marker；training 中 `
Answer:
` marker 存在且参与 loss。P0 ladder 原计划测试不预先提供 marker 的情况，但由于 direct-answer gate failed，ladder 未作为最终结论使用。

3. cache generation 是否与 full forward 一致？  
   一致。use_cache=True/False 与 full-forward first-step logits top-20 完全一致，max probability diff 约 1.67e-6。

4. direct-answer control 是否能自由 overfit？  
   不能。500 step 后 teacher-forced answer token acc=1.0，但 free answer acc=0.40625，valid answer rate=0.5625，未达到 0.95 gate。

5. prefix-forcing 几个 token 后结果开始恢复？  
   未完成。ladder 被停止以优先执行 direct-answer gate；由于 direct-answer 本身失败，当前应先修 generation/template/parser，而不是解释 latent transition。

6. O1/O2 是否解决 Forced-K 失败？  
   未运行。根据任务 gate，direct-answer control 未达 0.95 时必须停止 O1/O2。

7. 显式 boundary 是否必要？  
   现在不能下结论。O3 只应在 direct-answer 通过且 O1/O2 仍失败、并且 transition audit 指向边界问题时运行；当前 direct-answer 失败更优先。

8. 当前是否可以恢复 10k pilot？  
   不可以。

## Next Step

先修复或重新定义 free-generation protocol/parser/direct-answer setting。只有 direct-answer control 可以在同一 debug32 上 free overfit 到 >=0.95，才有资格继续 O1/O2 或 O3。
