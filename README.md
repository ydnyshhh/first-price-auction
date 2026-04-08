# first_price_auction

`first_price_auction` is a compact Prime / `verifiers` environment for sealed-bid first-price auctions. Each example gives the model a private value and a fully specified auction setting, asks for one bid, and scores that bid with deterministic Monte Carlo expected utility. No LLM judge is used.

## What It Tests

The environment targets strategic reasoning under incomplete information:

- first-price bid shading in textbook symmetric IPV settings
- adaptation to perturbed opponent policies
- robustness to value-distribution shift
- reasoning under reserve prices and tie rules

## Environment Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `num_examples` | `int` | `200` | Number of synthetic dataset rows to generate. |
| `seed` | `int` | `7` | Master seed for instance generation and rollout simulation. |
| `task_modes` | `tuple[str, ...]` | `("textbook", "perturbed_opponents", "distribution_shift", "reserve_price")` | Which task families to include. |
| `mode_weights` | `dict[str, float]` | `{}` | Optional weighted mode mix. Empty means balanced cycling. |
| `n_bidders_options` | `tuple[int, ...]` | `(2, 3, 4, 5)` | Possible total bidder counts. |
| `max_bid` | `float` | `100.0` | Maximum legal bid and default support upper bound. |
| `require_json_output` | `bool` | `True` | If true, prompts request `{"bid": ..., "reasoning_summary": ...}`. |
| `num_mc_samples` | `int` | `256` | Monte Carlo samples used to estimate expected utility. |
| `best_response_grid_size` | `int` | `41` | Grid size for approximate empirical best-response search. |
| `compute_best_response_baseline` | `bool` | `True` | Whether to compute best-response diagnostics. |
| `invalid_bid_penalty` | `float` | `-1.0` | Score used for parse failures or out-of-range bids. |
| `overbid_penalty` | `float` | `0.0` | Optional score penalty when `bid > private_value`. |
| `normalize_rewards` | `bool` | `False` | Logs `normalized_score = score / max(private_value, 1)`. |
| `tie_break_rule` | `str` | `"random"` | Either `"random"` or `"lose"`. |
| `reserve_price_fraction_range` | `tuple[float, float]` | `(0.08, 0.35)` | Reserve range, as a fraction of `max_bid`, for reserve-price tasks. |

## Scoring

For a valid submitted bid `b` with private value `v`, the environment simulates `num_mc_samples` auctions against the configured opponent simulator and estimates:

`expected_utility = E[(v - b) * 1{agent wins and clears reserve}]`

Reward behavior:

- invalid parse or out-of-range bid: `score = invalid_bid_penalty`
- otherwise: `score = expected_utility`
- if `overbid_penalty > 0`, the score is reduced when `bid > private_value`

The environment also computes:

- `reference_bid` and `reference_expected_utility`
- `best_response_bid` and `best_response_expected_utility` from an empirical bid grid search
- regret-style gaps against those baselines

## Logged Metrics

Numeric rollout metrics exposed through the rubric:

| Metric | Meaning |
| --- | --- |
| `score` | Primary environment reward. |
| `expected_utility` | Monte Carlo estimate of raw expected utility. |
| `submitted_bid` | Parsed bid, or `-1.0` if no numeric bid was parsed. |
| `bid_to_value_ratio` | `bid / private_value` when defined. |
| `win_rate_estimate` | Estimated probability of winning. |
| `overbid_flag` | `1.0` if `bid > private_value`, else `0.0`. |
| `parse_success` | `1.0` if a numeric bid was extracted, else `0.0`. |
| `json_valid` | `1.0` if strict JSON parsing succeeded with a usable `bid`. |
| `bid_in_range` | `1.0` if the parsed bid lies in `[0, max_bid]`. |
| `reference_bid` | Theoretical reference bid when available, else `-1.0`. |
| `reference_expected_utility` | Utility estimate for the reference bid, else `-1.0`. |
| `utility_gap_to_reference` | Submitted expected utility minus reference expected utility. |
| `best_response_bid` | Approximate empirical best response on the configured grid. Logged when `compute_best_response_baseline=True`. |
| `best_response_expected_utility` | Utility estimate of the best-response bid. Logged when `compute_best_response_baseline=True`. |
| `regret_to_best_response` | `best_response_expected_utility - expected_utility`, clipped at zero. Logged when `compute_best_response_baseline=True`. |
| `bid_error_to_best_response` | Absolute bid gap to the best-response bid when a bid is available. Logged when `compute_best_response_baseline=True`. |
| `normalized_score` | Optional normalized score. Logged when `normalize_rewards=True`. |
| `number_count` | How many numeric substrings were found during regex fallback. |
| `n_bidders` | Total bidder count as a numeric metric. |

Categorical auction metadata is stored in each example's `info` field and duplicated in dataset columns where useful, including `instance_id`, `task_mode`, `distribution_type`, `difficulty`, opponent-policy details, reserve price, and the simulation seed.

## Local Usage

```python
from auction_env import load_environment

env = load_environment(
    num_examples=8,
    num_mc_samples=64,
    compute_best_response_baseline=True,
)
```
