# Stage-1 Loss Weighting Analysis

CASE 2: O1/O2 improve over O0 but remain below gate; next should run explicit-boundary O3 if transition remains the bottleneck.

| experiment | kind | step | TF THINK | TF answer | Forced-K exact | Forced-K valid | Fixed-K exact | Free-K mean THINK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| O0 token-weighted | best | 500 | 0.9936239868402481 | 0.979761166498065 | 0.1875 | 0.28125 | 0.0 | 225.90625 |
| O1 segment-balanced | best | 400 | 0.982254995033145 | 0.9804035779088736 | 0.46875 | 0.53125 | 0.0 | None |
| O1 segment-balanced | final | 500 | 0.9991727396845818 | 0.9688841998577118 | 0.03125 | 0.09375 | 0.0 | 255.21875 |
| O2 answer-heavy | best | 450 | 0.8056109398603439 | 0.9961753059178591 | 0.75 | 0.84375 | 0.0 | None |
| O2 answer-heavy | final | 500 | 0.9686355292797089 | 0.9858491457998753 | 0.375 | 0.5 | 0.0 | 0.0 |

## Loss/Gradient Interpretation

Per-step O1/O2 logs include `think_weighted_contribution`, `answer_weighted_contribution`, `THINK_segment_gradient_norm`, and `answer_segment_gradient_norm`. Do not choose by total loss alone; use Forced-K and transition metrics.
