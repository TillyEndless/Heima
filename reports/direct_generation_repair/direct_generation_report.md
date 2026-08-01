# Direct Generation Repair Report

## Main Conclusion

The direct-answer checkpoint is much better than the old parsed accuracy suggested, but it still does **not** pass the >=0.95 free-generation gate.

- Correct shifted teacher-forced answer token accuracy: `0.9979661107063293`.
- Sequence-level teacher-forced exact match: `0.90625`.
- Canonical-prefix manual/generate token exact match: `0.90625`.
- Numeric parser accuracy: `0.59375`, but this undercounts because several gold answers are code/free text, not numeric answers.
- Gold-prefix induction mean accuracy: `0.9990972198549729`; sequence-all-correct rate: `0.90625`.

## Answers

1. 原 teacher-forced accuracy=1.0 是否计算正确？  
Mostly yes but not literally 1.0 under correct shifting: shifted answer token accuracy is `0.9979661107063293`, EOS accuracy is `0.96875`, sequence exact match is `0.90625`.

2. 是否存在 causal shift/off-by-one bug？  
No clear off-by-one bug in the new audit. Manual shifted CE matches model loss with abs diff `1.862645149230957e-09`.

3. train prefix 与 generation prefix 是否逐 token 一致？  
The old inference prefix is **not** identical: exact prefix match rate `0.0`, mismatch count `32/32`, answer marker difference count `31`. Canonical prefix must be derived from labels.

4. manual greedy 与 model.generate 是否一致？  
Manual vs generate(no-cache) exact-match rate `1.0`; manual vs generate(cache) exact-match rate `0.9375`. Generate no-cache/cache have the same aggregate token exact/parser rates.

5. token generation 与 parser 哪一层失败？  
Both layers matter, but parser is misleading for this dataset. Generate-cache token exact match is `0.90625`, while numeric parser accuracy is only `0.59375` because some correct token outputs are nonnumeric code/text answers.

6. direct-answer 32-sample 是否真正 free overfit？  
No. Using the primary token-exact criterion, it reaches `0.90625`, below 0.95.

7. 修复后旧 O0 是否通过 Forced-K？  
Skipped because direct-answer gate still does not pass.

8. 是否可以开始 O1/O2？  
No. O1/O2 remain blocked until direct-answer canonical-prefix token exact match reaches >=0.95 or the gate is explicitly changed.

9. 当前是否仍禁止 10k pilot 和 Stage2？  
Yes.

## Decision Tree

CASE 5 + parser applicability issue: old prefix was mismatched; canonical generation greatly improves token exactness, but direct-answer still falls short of the 0.95 gate. Fix direct-answer protocol/data/parser before any latent training.
