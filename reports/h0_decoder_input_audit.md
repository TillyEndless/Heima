# H0 Decoder Input Teacher-Forcing Audit

## Decoder Forward Location

- `scripts/run_data_small_vlm_official_sections.py::decoder_forward` constructs B-side inputs.
- `scripts/heima_alignment/ab_loss1_shortcut_formal.py::loss1_forward` calls it for `h0_heima_b_probe`.
- H0 uses `detach_encoder_latent=True`, A frozen, B/projectors trainable during training. This audit is offline only.

## Actual B Input Sequence

For each section, the input is:

`Question text + reconstruction instruction + <THINKING_OF_SECTION> latent slot + Target: + text_cot_i + EOS`

The latent slot token embedding is replaced with projected continuous latent before B forward via `inputs_embeds`; `attention_mask` is 1 for all non-padding tokens.

## Real Batch Token Dump

### summary

Question: `What was the population of the Dominican Republic in 2019? Answer the question using a single word or phrase.`

|pos|role|attention|label_active|token_id|token|label|prediction_source|
|-:|-|-:|-|-:|-|-|-:|
|0|prompt|1|False|14582|`Question`|`None`|None|
|1|prompt|1|False|510|`:\n`|`None`|None|
|2|prompt|1|False|3838|`What`|`None`|None|
|3|prompt|1|False|572|` was`|`None`|None|
|4|prompt|1|False|279|` the`|`None`|None|
|5|prompt|1|False|7042|` population`|`None`|None|
|6|prompt|1|False|315|` of`|`None`|None|
|7|prompt|1|False|279|` the`|`None`|None|
|8|prompt|1|False|66013|` Dominican`|`None`|None|
|9|prompt|1|False|5429|` Republic`|`None`|None|
|10|prompt|1|False|304|` in`|`None`|None|
|11|prompt|1|False|220|` `|`None`|None|
|12|prompt|1|False|17|`2`|`None`|None|
|13|prompt|1|False|15|`0`|`None`|None|
|14|prompt|1|False|16|`1`|`None`|None|
|15|prompt|1|False|24|`9`|`None`|None|
|16|prompt|1|False|30|`?`|`None`|None|
|17|prompt|1|False|21806|` Answer`|`None`|None|
|18|prompt|1|False|279|` the`|`None`|None|
|19|prompt|1|False|3405|` question`|`None`|None|
|20|prompt|1|False|1667|` using`|`None`|None|
|21|prompt|1|False|264|` a`|`None`|None|
|22|prompt|1|False|3175|` single`|`None`|None|
|23|prompt|1|False|3409|` word`|`None`|None|
|24|prompt|1|False|476|` or`|`None`|None|
|25|prompt|1|False|17133|` phrase`|`None`|None|
|26|prompt|1|False|382|`.\n\n`|`None`|None|
|27|prompt|1|False|16664|`Instruction`|`None`|None|
|28|prompt|1|False|510|`:\n`|`None`|None|
|29|prompt|1|False|693|`Re`|`None`|None|
|30|prompt|1|False|7596|`construct`|`None`|None|
|31|prompt|1|False|279|` the`|`None`|None|
|32|prompt|1|False|1260|` He`|`None`|None|
|33|prompt|1|False|7523|`ima`|`None`|None|
|34|prompt|1|False|12126|` summary`|`None`|None|
|35|prompt|1|False|3381|` thought`|`None`|None|
|36|prompt|1|False|504|` from`|`None`|None|
|37|prompt|1|False|279|` the`|`None`|None|
|38|prompt|1|False|41667|` latent`|`None`|None|
|39|prompt|1|False|13|`.`|`None`|None|
|40|prompt|1|False|3155|` Do`|`None`|None|
|41|prompt|1|False|537|` not`|`None`|None|
|42|prompt|1|False|990|` use`|`None`|None|
|43|prompt|1|False|279|` the`|`None`|None|
|44|prompt|1|False|2168|` image`|`None`|None|
|45|prompt|1|False|382|`.\n\n`|`None`|None|
|46|latent_slot|1|False|151665|`<THINKING_OF_SUMMARY>`|`None`|None|
|47|prompt|1|False|271|`\n\n`|`None`|None|
|48|prompt|1|False|6397|`Target`|`None`|None|
|49|prompt|1|False|510|`:\n`|`None`|None|
|50|target_text_cot|1|True|1249|`To`|`To`|49|
|51|target_text_cot|1|True|1477|` find`|` find`|50|
|52|target_text_cot|1|True|279|` the`|` the`|51|
|53|target_text_cot|1|True|7042|` population`|` population`|52|
|54|target_text_cot|1|True|315|` of`|` of`|53|
|55|target_text_cot|1|True|279|` the`|` the`|54|
|56|target_text_cot|1|True|66013|` Dominican`|` Dominican`|55|
|57|target_text_cot|1|True|5429|` Republic`|` Republic`|56|
|58|target_text_cot|1|True|304|` in`|` in`|57|
|59|target_text_cot|1|True|220|` `|` `|58|
|60|target_text_cot|1|True|17|`2`|`2`|59|
|61|target_text_cot|1|True|15|`0`|`0`|60|
|62|target_text_cot|1|True|16|`1`|`1`|61|
|63|target_text_cot|1|True|24|`9`|`9`|62|
|64|target_text_cot|1|True|11|`,`|`,`|63|
|65|target_text_cot|1|True|358|` I`|` I`|64|
|66|target_text_cot|1|True|686|` will`|` will`|65|
|67|target_text_cot|1|True|23643|` analyze`|` analyze`|66|
|68|target_text_cot|1|True|279|` the`|` the`|67|
|69|target_text_cot|1|True|3619|` bar`|` bar`|68|
|70|target_text_cot|1|True|9487|` chart`|` chart`|69|
|71|target_text_cot|1|True|3897|` provided`|` provided`|70|
|72|target_text_cot|1|True|304|` in`|` in`|71|
|73|target_text_cot|1|True|279|` the`|` the`|72|
|74|target_text_cot|1|True|2168|` image`|` image`|73|
|75|target_text_cot|1|True|11|`,`|`,`|74|
|76|target_text_cot|1|True|21080|` focusing`|` focusing`|75|
|77|target_text_cot|1|True|389|` on`|` on`|76|
|78|target_text_cot|1|True|279|` the`|` the`|77|
|79|target_text_cot|1|True|3619|` bar`|` bar`|78|

Latent slot positions: `[46]`; prompt length: `50`; target length: `41`.

### caption

Question: `What was the population of the Dominican Republic in 2019? Answer the question using a single word or phrase.`

|pos|role|attention|label_active|token_id|token|label|prediction_source|
|-:|-|-:|-|-:|-|-|-:|
|0|prompt|1|False|14582|`Question`|`None`|None|
|1|prompt|1|False|510|`:\n`|`None`|None|
|2|prompt|1|False|3838|`What`|`None`|None|
|3|prompt|1|False|572|` was`|`None`|None|
|4|prompt|1|False|279|` the`|`None`|None|
|5|prompt|1|False|7042|` population`|`None`|None|
|6|prompt|1|False|315|` of`|`None`|None|
|7|prompt|1|False|279|` the`|`None`|None|
|8|prompt|1|False|66013|` Dominican`|`None`|None|
|9|prompt|1|False|5429|` Republic`|`None`|None|
|10|prompt|1|False|304|` in`|`None`|None|
|11|prompt|1|False|220|` `|`None`|None|
|12|prompt|1|False|17|`2`|`None`|None|
|13|prompt|1|False|15|`0`|`None`|None|
|14|prompt|1|False|16|`1`|`None`|None|
|15|prompt|1|False|24|`9`|`None`|None|
|16|prompt|1|False|30|`?`|`None`|None|
|17|prompt|1|False|21806|` Answer`|`None`|None|
|18|prompt|1|False|279|` the`|`None`|None|
|19|prompt|1|False|3405|` question`|`None`|None|
|20|prompt|1|False|1667|` using`|`None`|None|
|21|prompt|1|False|264|` a`|`None`|None|
|22|prompt|1|False|3175|` single`|`None`|None|
|23|prompt|1|False|3409|` word`|`None`|None|
|24|prompt|1|False|476|` or`|`None`|None|
|25|prompt|1|False|17133|` phrase`|`None`|None|
|26|prompt|1|False|382|`.\n\n`|`None`|None|
|27|prompt|1|False|16664|`Instruction`|`None`|None|
|28|prompt|1|False|510|`:\n`|`None`|None|
|29|prompt|1|False|693|`Re`|`None`|None|
|30|prompt|1|False|7596|`construct`|`None`|None|
|31|prompt|1|False|279|` the`|`None`|None|
|32|prompt|1|False|1260|` He`|`None`|None|
|33|prompt|1|False|7523|`ima`|`None`|None|
|34|prompt|1|False|17256|` caption`|`None`|None|
|35|prompt|1|False|3381|` thought`|`None`|None|
|36|prompt|1|False|504|` from`|`None`|None|
|37|prompt|1|False|279|` the`|`None`|None|
|38|prompt|1|False|41667|` latent`|`None`|None|
|39|prompt|1|False|13|`.`|`None`|None|
|40|prompt|1|False|3155|` Do`|`None`|None|
|41|prompt|1|False|537|` not`|`None`|None|
|42|prompt|1|False|990|` use`|`None`|None|
|43|prompt|1|False|279|` the`|`None`|None|
|44|prompt|1|False|2168|` image`|`None`|None|
|45|prompt|1|False|382|`.\n\n`|`None`|None|
|46|latent_slot|1|False|151666|`<THINKING_OF_CAPTION>`|`None`|None|
|47|prompt|1|False|271|`\n\n`|`None`|None|
|48|prompt|1|False|6397|`Target`|`None`|None|
|49|prompt|1|False|510|`:\n`|`None`|None|
|50|target_text_cot|1|True|785|`The`|`The`|49|
|51|target_text_cot|1|True|2168|` image`|` image`|50|
|52|target_text_cot|1|True|374|` is`|` is`|51|
|53|target_text_cot|1|True|264|` a`|` a`|52|
|54|target_text_cot|1|True|3619|` bar`|` bar`|53|
|55|target_text_cot|1|True|9487|` chart`|` chart`|54|
|56|target_text_cot|1|True|27940|` displaying`|` displaying`|55|
|57|target_text_cot|1|True|279|` the`|` the`|56|
|58|target_text_cot|1|True|7042|` population`|` population`|57|
|59|target_text_cot|1|True|315|` of`|` of`|58|
|60|target_text_cot|1|True|279|` the`|` the`|59|
|61|target_text_cot|1|True|66013|` Dominican`|` Dominican`|60|
|62|target_text_cot|1|True|5429|` Republic`|` Republic`|61|
|63|target_text_cot|1|True|304|` in`|` in`|62|
|64|target_text_cot|1|True|11728|` millions`|` millions`|63|
|65|target_text_cot|1|True|504|` from`|` from`|64|
|66|target_text_cot|1|True|279|` the`|` the`|65|
|67|target_text_cot|1|True|1635|` years`|` years`|66|
|68|target_text_cot|1|True|220|` `|` `|67|
|69|target_text_cot|1|True|17|`2`|`2`|68|
|70|target_text_cot|1|True|15|`0`|`0`|69|
|71|target_text_cot|1|True|16|`1`|`1`|70|
|72|target_text_cot|1|True|21|`6`|`6`|71|
|73|target_text_cot|1|True|311|` to`|` to`|72|
|74|target_text_cot|1|True|220|` `|` `|73|
|75|target_text_cot|1|True|17|`2`|`2`|74|
|76|target_text_cot|1|True|15|`0`|`0`|75|
|77|target_text_cot|1|True|17|`2`|`2`|76|
|78|target_text_cot|1|True|21|`6`|`6`|77|
|79|target_text_cot|1|True|13|`.`|`.`|78|

Latent slot positions: `[46]`; prompt length: `50`; target length: `51`.

### reasoning

Question: `What was the population of the Dominican Republic in 2019? Answer the question using a single word or phrase.`

|pos|role|attention|label_active|token_id|token|label|prediction_source|
|-:|-|-:|-|-:|-|-|-:|
|0|prompt|1|False|14582|`Question`|`None`|None|
|1|prompt|1|False|510|`:\n`|`None`|None|
|2|prompt|1|False|3838|`What`|`None`|None|
|3|prompt|1|False|572|` was`|`None`|None|
|4|prompt|1|False|279|` the`|`None`|None|
|5|prompt|1|False|7042|` population`|`None`|None|
|6|prompt|1|False|315|` of`|`None`|None|
|7|prompt|1|False|279|` the`|`None`|None|
|8|prompt|1|False|66013|` Dominican`|`None`|None|
|9|prompt|1|False|5429|` Republic`|`None`|None|
|10|prompt|1|False|304|` in`|`None`|None|
|11|prompt|1|False|220|` `|`None`|None|
|12|prompt|1|False|17|`2`|`None`|None|
|13|prompt|1|False|15|`0`|`None`|None|
|14|prompt|1|False|16|`1`|`None`|None|
|15|prompt|1|False|24|`9`|`None`|None|
|16|prompt|1|False|30|`?`|`None`|None|
|17|prompt|1|False|21806|` Answer`|`None`|None|
|18|prompt|1|False|279|` the`|`None`|None|
|19|prompt|1|False|3405|` question`|`None`|None|
|20|prompt|1|False|1667|` using`|`None`|None|
|21|prompt|1|False|264|` a`|`None`|None|
|22|prompt|1|False|3175|` single`|`None`|None|
|23|prompt|1|False|3409|` word`|`None`|None|
|24|prompt|1|False|476|` or`|`None`|None|
|25|prompt|1|False|17133|` phrase`|`None`|None|
|26|prompt|1|False|382|`.\n\n`|`None`|None|
|27|prompt|1|False|16664|`Instruction`|`None`|None|
|28|prompt|1|False|510|`:\n`|`None`|None|
|29|prompt|1|False|693|`Re`|`None`|None|
|30|prompt|1|False|7596|`construct`|`None`|None|
|31|prompt|1|False|279|` the`|`None`|None|
|32|prompt|1|False|1260|` He`|`None`|None|
|33|prompt|1|False|7523|`ima`|`None`|None|
|34|prompt|1|False|32711|` reasoning`|`None`|None|
|35|prompt|1|False|3381|` thought`|`None`|None|
|36|prompt|1|False|504|` from`|`None`|None|
|37|prompt|1|False|279|` the`|`None`|None|
|38|prompt|1|False|41667|` latent`|`None`|None|
|39|prompt|1|False|13|`.`|`None`|None|
|40|prompt|1|False|3155|` Do`|`None`|None|
|41|prompt|1|False|537|` not`|`None`|None|
|42|prompt|1|False|990|` use`|`None`|None|
|43|prompt|1|False|279|` the`|`None`|None|
|44|prompt|1|False|2168|` image`|`None`|None|
|45|prompt|1|False|382|`.\n\n`|`None`|None|
|46|latent_slot|1|False|151667|`<THINKING_OF_REASONING>`|`None`|None|
|47|prompt|1|False|271|`\n\n`|`None`|None|
|48|prompt|1|False|6397|`Target`|`None`|None|
|49|prompt|1|False|510|`:\n`|`None`|None|
|50|target_text_cot|1|True|1249|`To`|`To`|49|
|51|target_text_cot|1|True|8253|` determine`|` determine`|50|
|52|target_text_cot|1|True|279|` the`|` the`|51|
|53|target_text_cot|1|True|7042|` population`|` population`|52|
|54|target_text_cot|1|True|369|` for`|` for`|53|
|55|target_text_cot|1|True|220|` `|` `|54|
|56|target_text_cot|1|True|17|`2`|`2`|55|
|57|target_text_cot|1|True|15|`0`|`0`|56|
|58|target_text_cot|1|True|16|`1`|`1`|57|
|59|target_text_cot|1|True|24|`9`|`9`|58|
|60|target_text_cot|1|True|11|`,`|`,`|59|
|61|target_text_cot|1|True|358|` I`|` I`|60|
|62|target_text_cot|1|True|686|` will`|` will`|61|
|63|target_text_cot|1|True|24523|` locate`|` locate`|62|
|64|target_text_cot|1|True|279|` the`|` the`|63|
|65|target_text_cot|1|True|3619|` bar`|` bar`|64|
|66|target_text_cot|1|True|29829|` labeled`|` labeled`|65|
|67|target_text_cot|1|True|448|` with`|` with`|66|
|68|target_text_cot|1|True|279|` the`|` the`|67|
|69|target_text_cot|1|True|1042|` year`|` year`|68|
|70|target_text_cot|1|True|220|` `|` `|69|
|71|target_text_cot|1|True|17|`2`|`2`|70|
|72|target_text_cot|1|True|15|`0`|`0`|71|
|73|target_text_cot|1|True|16|`1`|`1`|72|
|74|target_text_cot|1|True|24|`9`|`9`|73|
|75|target_text_cot|1|True|13|`.`|`.`|74|
|76|target_text_cot|1|True|10548|` According`|` According`|75|
|77|target_text_cot|1|True|311|` to`|` to`|76|
|78|target_text_cot|1|True|279|` the`|` the`|77|
|79|target_text_cot|1|True|3619|` bar`|` bar`|78|

Latent slot positions: `[46]`; prompt length: `50`; target length: `56`.

## Label Mask

Labels are `-100` for the prompt, including question/instruction/latent slot. Labels are active only for target CoT tokens plus EOS. `heima_ce_loss` shifts labels, so label at position `p` is predicted from logits at `p-1`.

## Causal Visibility

### summary

- Status: `PASS_NO_CAUSAL_LEAK_FOUND`
- Prompt length: `50`
- Target length: `41`
- Latent slot positions: `[46]`
- Active label count: `41`

First active-label checks show no future target positions visible to their prediction source.

### caption

- Status: `PASS_NO_CAUSAL_LEAK_FOUND`
- Prompt length: `50`
- Target length: `51`
- Latent slot positions: `[46]`
- Active label count: `51`

First active-label checks show no future target positions visible to their prediction source.

### reasoning

- Status: `PASS_NO_CAUSAL_LEAK_FOUND`
- Prompt length: `50`
- Target length: `56`
- Latent slot positions: `[46]`
- Active label count: `56`

First active-label checks show no future target positions visible to their prediction source.

## Text Corruption Test

- `summary`: `PASS_NO_FUTURE_TEXT_LEAKAGE`, max prefix loss diff `0`, mean suffix diff `3.9896`
- `caption`: `PASS_NO_FUTURE_TEXT_LEAKAGE`, max prefix loss diff `0`, mean suffix diff `2.6064`
- `reasoning`: `PASS_NO_FUTURE_TEXT_LEAKAGE`, max prefix loss diff `0`, mean suffix diff `2.17165`

## Latent Gradient

- `summary`: norm `1.39223e-06`, abs_mean `2.47078e-08`, abs_max `1.06636e-07`
- `caption`: norm `4.47922e-06`, abs_mean `7.84494e-08`, abs_max `3.76254e-07`
- `reasoning`: norm `2.04515e-06`, abs_mean `3.61532e-08`, abs_max `1.44355e-07`

## Intervention Replay

|section|NLL_correct|NLL_shuffle|NLL_qonly|NLL_zero|shuffle_margin|shuffle CI|
|-|-:|-:|-:|-:|-:|-|
|summary|1.197146|1.197249|1.198058|1.197047|0.00010262|[-0.00016617710934951901, 0.0003810862544924021]|
|caption|1.466847|1.466682|1.467311|1.466598|-0.00016428|[-0.000452160369604826, 9.678385686129332e-05]|
|reasoning|1.460980|1.461046|1.461390|1.461305|0.00006598|[-0.0001726023037917912, 0.00029111094772815704]|
|avg|1.374991|1.374992|1.375586|1.374983|0.00000144|-|

## Final Answer

H0 is equivalent to `P(text_cot_i | question, latent, text_cot_prefix)` under standard causal teacher forcing. The `text_cot_prefix` consists only of previously generated gold target tokens and is legal teacher-forcing history. Future target tokens are not visible to the prediction source positions.

No causal future-text leakage was found.
