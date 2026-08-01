# Length And Cap Audit

{
  "question_tokens": {
    "P50": 64.0,
    "P75": 103.0,
    "P90": 180.10000000000002,
    "P95": 301.14999999999986,
    "P99": 796.2999999999997
  },
  "gold_cot_tokens": {
    "P50": 870.0,
    "P75": 1737.75,
    "P90": 4054.5000000000014,
    "P95": 6107.149999999996,
    "P99": 11997.57
  },
  "answer_tokens": {
    "P50": 478.0,
    "P75": 683.5,
    "P90": 874.0,
    "P95": 979.05,
    "P99": 1324.209999999999
  },
  "raw_K": {
    "P50": 435.0,
    "P75": 869.25,
    "P90": 2027.3000000000006,
    "P95": 3054.0499999999984,
    "P99": 5998.29
  },
  "used_K": {
    "P50": 128.0,
    "P75": 128.0,
    "P90": 128.0,
    "P95": 128.0,
    "P99": 128.0
  },
  "effective_ratio": {
    "P50": 0.1471264367816092,
    "P75": 0.22348332295976275,
    "P90": 0.30403800475059384,
    "P95": 0.38232042205888966,
    "P99": 0.5
  },
  "stage1_sequence_length": {
    "P50": 316.0,
    "P75": 354.25,
    "P90": 419.1,
    "P95": 477.04999999999995,
    "P99": 512.0
  },
  "stage2_decode_sequence_length": {
    "P50": 447.0,
    "P75": 487.0,
    "P90": 560.2,
    "P95": 640.0,
    "P99": 640.0
  },
  "latent_cap_rate": 0.979,
  "question_truncation_rate": 0.063,
  "cot_truncation_rate": 0.981,
  "answer_truncation_rate": 0.841,
  "protocol_notes": {
    "raw_r_0p5": "raw_K=round(0.5*gold_cot_tokens)",
    "capped_r_0p5": "used_K=min(raw_K,128) in current smoke/gates",
    "fixed_K": "not used here",
    "oracle_K": "raw_K uses gold CoT length and is diagnostic only",
    "free_K": "not used here"
  }
}
