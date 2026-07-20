# HEIMA-OVERLAP-AUDIT

Status: audit-only; no code, config, data, checkpoint, retraining, G2, or Loss2 changes were made.

## Official Commit

- SHA: `cebd6ef3ef862f62c601ced191b64b45d2bf3de2`
- Subject: `update README.md`
- Author date: `2026-05-20T11:51:29+08:00`
- Remote: `git@github.com:TillyEndless/Heima.git`

## Audited Files

Official Heima files: 13
Current G1 files/config/data: 8
Full lists are recorded in `heima_official_spec.json` and `current_g1_spec.json`.

## Matrix Stats

- Items: 40
- FUNCTIONALLY_ALIGNED: 5
- PARTIAL: 6
- NOT_ALIGNED: 29
- HEIMA_BASELINE_REQUIRED rows: 27
- RESOURCE_ADAPTATION rows: 10
- INTENDED_METHOD_DIFFERENCE rows: 3
- Critical mismatches: 14

## Critical Mismatches

- M04 required answer sections: official=Requires SUMMARY/CAPTION/REASONING extraction; current=No section extraction; classification=HEIMA_BASELINE_REQUIRED.
- M05 main compressed answer: official=Replaces section spans with typed thinking tokens; current=No replacement; z inserted as hidden vector; classification=HEIMA_BASELINE_REQUIRED.
- M06 thinking token types: official=Three typed tokens; current=Single latent vector; EOS placeholder only in B prompt; classification=HEIMA_BASELINE_REQUIRED.
- M08 decoder datasets: official=Three local decoder datasets; current=One whole-CoT decoder dataset; classification=HEIMA_BASELINE_REQUIRED.
- M09 decoder user prompt includes question: official=Decoder templates include original question; current=B prompt omits question; classification=HEIMA_BASELINE_REQUIRED.
- M10 decoder target scope: official=Local tagged section; current=Whole CoT text; classification=HEIMA_BASELINE_REQUIRED.
- M13 thinking token supervised: official=Compressed assistant contains typed thinking tokens; current=No typed thinking token target; classification=HEIMA_BASELINE_REQUIRED.
- M14 latent extraction site: official=Hidden states at shifted typed thinking-token positions; current=Last ordinary question token hidden; classification=HEIMA_BASELINE_REQUIRED.
- M16 abstract projection: official=Decoder has abstract_projection for latent tokens; current=No projector; direct hidden replacement; classification=HEIMA_BASELINE_REQUIRED.
- M17 decoder count: official=Three decoders summary/caption/reasoning; current=One decoder B; classification=HEIMA_BASELINE_REQUIRED.
- M28 special token registration: official=Tokenizer has explicit <THINKING_OF_*> special IDs; current=Uses GPT-2 EOS as placeholder; classification=HEIMA_BASELINE_REQUIRED.
- M36 latent semantic pressure: official=Typed local decoders force section-specific semantics; current=Whole-CoT decoder permits global/generic pressure; classification=HEIMA_BASELINE_REQUIRED.
- M37 A output surface: official=A is trained in normal LM token space with visible thinking tokens; current=A is trained with hidden z insertion and answer only; classification=HEIMA_BASELINE_REQUIRED.
- M40 lightweight Heima baseline claim: official=Would preserve section tokens, local decoders, projector, question-conditioned decoder; current=Current preserves only joint no-detach latent-to-A idea and CE scaffolding; classification=HEIMA_BASELINE_REQUIRED.

## Can Current G1 Be Called a Lightweight Heima Baseline?

No. The current G1 should be called a Heima-inspired latent-reasoning probe, not a lightweight Heima baseline. It preserves a small subset of method intent: a main loss, a decoder loss, no detach on the latent, Loss1 returning gradient to A through B, and a joint objective. It does not preserve the baseline-required Heima overlap structure: typed summary/caption/reasoning thinking tokens, section-replacement data, question-conditioned local decoders, abstract projection, progressive stage variants, or multimodal VLM data.

## Resource Adaptations

Resource adaptations are valid operational choices for the remote environment: GPT-2 instead of Llama-3.2-Vision-11B, text-only synthetic data, single-process training instead of FSDP, local_files_only model loading, simplified checkpoints, and full small-model finetuning. These adaptations should be labeled as such and not confused with Heima-baseline equivalence.

## Intended Differences

Only three intended method differences are accepted in this audit and are recorded in `intended_differences.json`: Loss1 does not detach latent, Loss1 through B returns gradients to A, and Main + lambda1 * Loss1 joint optimization. Current choices outside that list, especially omitting the question from B and decoding whole CoT instead of local sections, are classified as HEIMA_BASELINE_REQUIRED deviations.

## Accidental Deviations

The accidental/baseline-required deviations are listed in `accidental_deviations.json`. The highest-impact deviations for the observed normal-vs-shuffled whole-CoT NLL overlap are: current z is an ordinary final question-token hidden state, B sees no question, B decodes whole CoT rather than local Heima sections, there are no typed thinking tokens, and there is no abstract projection.

## Tensor Trace Answer

`current_g1_tensor_trace.json` records that current G1 z is extracted by `last_valid_hidden` over a question-only A forward. It is therefore an ordinary question-token hidden state, not a special thinking-token hidden state. B receives a fixed instruction with an EOS placeholder replaced by z; B does not receive the question.
