# Heima Decoder Shortcut Final Analysis

## official_scaled_h0

### Prompt-only / Generation

- `correct`: BLEU1 `0.3911`, ROUGE-L `0.3297`, answer substring acc `0.0833`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`
- `prompt_only`: BLEU1 `0.3582`, ROUGE-L `0.3055`, answer substring acc `0.0833`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`
- `shuffle`: BLEU1 `0.3862`, ROUGE-L `0.3247`, answer substring acc `0.0833`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`
- `zero`: BLEU1 `0.3397`, ROUGE-L `0.3028`, answer substring acc `0.1667`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`

### Four-way NLL

|section|Q+correct|Q+shuffle|Q only|Z only|latent gain|shuffle margin|question shortcut ratio|
|-|-:|-:|-:|-:|-:|-:|-:|
|summary|1.586204|1.586587|1.588062|2.447367|0.00185878|0.00038349|-0.002163|
|caption|1.746829|1.746899|1.736756|2.131676|-0.01007356|0.00006913|0.025508|
|reasoning|1.725188|1.724956|1.708482|2.162350|-0.01670540|-0.00023176|0.036807|
|avg|1.686074|1.686147|1.677767|2.247131|-0.00830673|0.00007362|0.014589|

### Text Prefix Corruption

- `summary`: replace `10` prefix tokens, suffix loss delta `0.947799`
- `caption`: replace `12` prefix tokens, suffix loss delta `0.694411`
- `reasoning`: replace `12` prefix tokens, suffix loss delta `0.392424`

### Judgment

- Average latent gain is `-0.00830673` and average shuffle margin is `0.00007362`.
- The decoder uses legal teacher-forcing text prefix; prefix corruption changes later-token loss, so reconstruction is not a pure `P(text | question, latent)` sequence-level test.
- Correct latent is not clearly better than shuffled latent; evidence favors decoder/question/prefix shortcut over robust sample-specific latent use.

## our_h0

### Prompt-only / Generation

- `correct`: BLEU1 `0.3999`, ROUGE-L `0.3556`, answer substring acc `0.1250`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`
- `prompt_only`: BLEU1 `0.3641`, ROUGE-L `0.3251`, answer substring acc `0.0833`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`
- `shuffle`: BLEU1 `0.4043`, ROUGE-L `0.3582`, answer substring acc `0.1250`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`
- `zero`: BLEU1 `0.4070`, ROUGE-L `0.3603`, answer substring acc `0.0833`, BERTScore `{'available': False, 'reason': "No module named 'bert_score'"}`

### Four-way NLL

|section|Q+correct|Q+shuffle|Q only|Z only|latent gain|shuffle margin|question shortcut ratio|
|-|-:|-:|-:|-:|-:|-:|-:|
|summary|1.176712|1.176486|1.177121|1.995009|0.00040828|-0.00022692|-0.000499|
|caption|1.483021|1.482983|1.483589|1.840109|0.00056828|-0.00003729|-0.001594|
|reasoning|1.510951|1.510703|1.510935|1.894158|-0.00001622|-0.00024743|0.000042|
|avg|1.390228|1.390057|1.390548|1.909759|0.00032011|-0.00017055|-0.000617|

### Text Prefix Corruption

- `summary`: replace `10` prefix tokens, suffix loss delta `0.879365`
- `caption`: replace `12` prefix tokens, suffix loss delta `0.784126`
- `reasoning`: replace `12` prefix tokens, suffix loss delta `0.484561`

### Judgment

- Average latent gain is `0.00032011` and average shuffle margin is `-0.00017055`.
- The decoder uses legal teacher-forcing text prefix; prefix corruption changes later-token loss, so reconstruction is not a pure `P(text | question, latent)` sequence-level test.
- Correct latent is not clearly better than shuffled latent; evidence favors decoder/question/prefix shortcut over robust sample-specific latent use.

## Required Answers

1. The prompt contains a strong structural prior: question text, section name, reconstruction instruction, and teacher-forced CoT prefix during loss.
2. Question-only reconstruction is close to full latent-conditioned reconstruction in NLL and generation similarity.
3. Latent-only does not provide strong sample-specific reconstruction evidence in this audit.
4. Correct latent is not meaningfully better than shuffled latent; margins are near zero.
5. If shuffle margin is near zero, the best-supported explanation is mainly B/C: decoder shortcut via question/prompt/teacher-forced prefix plus prompt design. A cannot be ruled out, but latent geometry alone was nonzero in prior metrics, so lack of use by B is the immediate failure mode. D is less supported because the same dataset fields/images are complete in manifests.
