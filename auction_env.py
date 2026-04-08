"""Single-turn sealed-bid first-price auction environment for Prime/verifiers.

The environment measures whether a model can choose strategically sensible bids
when given its private value and a compact description of the auction setting.
Scoring is exact with respect to the simulator used by the environment: the
submitted bid is parsed, evaluated with deterministic Monte Carlo rollouts, and
compared to a baseline reference and an empirical best response.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import random
import re
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from datasets import Dataset
    import verifiers as vf

JsonDict = dict[str, Any]
Scenario = dict[str, Any]

TASK_MODES = (
    "textbook",
    "perturbed_opponents",
    "distribution_shift",
    "reserve_price",
)
TIE_BREAK_RULES = ("random", "lose")
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
EPSILON = 1e-9
MISSING_FLOAT = -1.0


@dataclass(frozen=True)
class AuctionEnvConfig:
    """Configuration for the first-price auction environment."""

    num_examples: int = 200
    seed: int = 7
    task_modes: tuple[str, ...] = TASK_MODES
    mode_weights: dict[str, float] = field(default_factory=dict)
    n_bidders_options: tuple[int, ...] = (2, 3, 4, 5)
    max_bid: float = 100.0
    require_json_output: bool = True
    num_mc_samples: int = 256
    best_response_grid_size: int = 41
    compute_best_response_baseline: bool = True
    invalid_bid_penalty: float = -1.0
    overbid_penalty: float = 0.0
    normalize_rewards: bool = False
    tie_break_rule: str = "random"
    reserve_price_fraction_range: tuple[float, float] = (0.08, 0.35)

    def __post_init__(self) -> None:
        if self.num_examples <= 0:
            raise ValueError("num_examples must be positive")
        if self.num_mc_samples <= 0:
            raise ValueError("num_mc_samples must be positive")
        if self.best_response_grid_size < 2:
            raise ValueError("best_response_grid_size must be at least 2")
        if self.max_bid <= 0:
            raise ValueError("max_bid must be positive")
        if not self.task_modes:
            raise ValueError("task_modes must not be empty")
        if any(mode not in TASK_MODES for mode in self.task_modes):
            raise ValueError(f"task_modes must be drawn from {TASK_MODES}")
        if not self.n_bidders_options or any(n < 2 for n in self.n_bidders_options):
            raise ValueError("n_bidders_options must contain integers >= 2")
        if self.tie_break_rule not in TIE_BREAK_RULES:
            raise ValueError(f"tie_break_rule must be one of {TIE_BREAK_RULES}")
        reserve_low, reserve_high = self.reserve_price_fraction_range
        if not (0.0 <= reserve_low <= reserve_high <= 1.0):
            raise ValueError("reserve_price_fraction_range must lie in [0, 1]")


@dataclass(frozen=True)
class ParseResult:
    """Parsed bid plus explicit diagnostics."""

    bid: float | None
    parse_success: bool
    json_valid: bool
    bid_in_range: bool
    parse_mode: str
    parse_error: str
    number_count: int


def round_amount(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def clip_value(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def build_mode_schedule(config: AuctionEnvConfig) -> list[str]:
    modes = list(config.task_modes)
    if not config.mode_weights:
        return [modes[i % len(modes)] for i in range(config.num_examples)]

    weights = [max(config.mode_weights.get(mode, 0.0), 0.0) for mode in modes]
    total = sum(weights)
    if total <= 0:
        return [modes[i % len(modes)] for i in range(config.num_examples)]

    ideal = [config.num_examples * weight / total for weight in weights]
    counts = [int(value) for value in ideal]
    remainder = config.num_examples - sum(counts)
    ranked = sorted(
        range(len(modes)),
        key=lambda idx: (ideal[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in ranked[:remainder]:
        counts[idx] += 1

    schedule: list[str] = []
    for mode, count in zip(modes, counts, strict=True):
        schedule.extend([mode] * count)

    random.Random(config.seed + 17).shuffle(schedule)
    return schedule


def sample_distribution_spec(
    task_mode: str,
    rng: random.Random,
    max_bid: float,
) -> tuple[str, JsonDict]:
    if task_mode == "textbook":
        return "uniform", {"low": 0.0, "high": max_bid}

    if task_mode == "perturbed_opponents":
        return "uniform", {"low": 0.0, "high": max_bid}

    if task_mode == "reserve_price":
        if rng.random() < 0.6:
            return "uniform", {"low": 0.0, "high": max_bid}
        mode = round_amount(rng.uniform(0.25 * max_bid, 0.8 * max_bid))
        return "triangular", {"low": 0.0, "high": max_bid, "mode": mode}

    draw = rng.random()
    if draw < 0.34:
        return "uniform", {"low": 0.0, "high": max_bid}
    if draw < 0.67:
        mode = round_amount(rng.uniform(0.15 * max_bid, 0.85 * max_bid))
        return "triangular", {"low": 0.0, "high": max_bid, "mode": mode}

    scale = max_bid / 100.0
    supports = [
        {
            "name": "mid_heavy",
            "points": [
                {"value": round_amount(5.0 * scale), "prob": 0.10},
                {"value": round_amount(25.0 * scale), "prob": 0.20},
                {"value": round_amount(50.0 * scale), "prob": 0.35},
                {"value": round_amount(75.0 * scale), "prob": 0.20},
                {"value": round_amount(100.0 * scale), "prob": 0.15},
            ],
        },
        {
            "name": "coarse_four_point",
            "points": [
                {"value": round_amount(10.0 * scale), "prob": 0.25},
                {"value": round_amount(30.0 * scale), "prob": 0.25},
                {"value": round_amount(60.0 * scale), "prob": 0.25},
                {"value": round_amount(90.0 * scale), "prob": 0.25},
            ],
        },
        {
            "name": "top_heavy",
            "points": [
                {"value": 0.0, "prob": 0.05},
                {"value": round_amount(20.0 * scale), "prob": 0.10},
                {"value": round_amount(45.0 * scale), "prob": 0.20},
                {"value": round_amount(70.0 * scale), "prob": 0.30},
                {"value": round_amount(100.0 * scale), "prob": 0.35},
            ],
        },
    ]
    return "discrete", supports[rng.randrange(len(supports))]


def sample_value(rng: random.Random, distribution_type: str, params: JsonDict) -> float:
    if distribution_type == "uniform":
        return rng.uniform(float(params["low"]), float(params["high"]))
    if distribution_type == "triangular":
        return rng.triangular(
            float(params["low"]),
            float(params["high"]),
            float(params["mode"]),
        )
    if distribution_type == "discrete":
        threshold = rng.random()
        cumulative = 0.0
        chosen = params["points"][-1]["value"]
        for point in params["points"]:
            cumulative += float(point["prob"])
            chosen = point["value"]
            if threshold <= cumulative:
                break
        return float(chosen)
    raise ValueError(f"Unsupported distribution_type: {distribution_type}")


def describe_distribution(distribution_type: str, params: JsonDict) -> str:
    if distribution_type == "uniform":
        return f"Uniform[{params['low']}, {params['high']}]"
    if distribution_type == "triangular":
        return (
            "Triangular distribution on "
            f"[{params['low']}, {params['high']}] with mode {params['mode']}"
        )
    points = ", ".join(f"{point['value']} ({point['prob']:.2f})" for point in params["points"])
    return f"Discrete distribution '{params['name']}' with support {{{points}}}"


def sample_policy_spec(
    task_mode: str,
    rng: random.Random,
    n_bidders: int,
    max_bid: float,
) -> tuple[str, JsonDict]:
    if task_mode == "textbook":
        return "equilibrium", {"alpha": round_amount((n_bidders - 1) / n_bidders, 4)}

    if task_mode == "perturbed_opponents":
        draw = rng.random()
        if draw < 0.25:
            return "truthful", {}
        if draw < 0.55:
            return "fractional", {"alpha": round_amount(rng.uniform(0.55, 0.9), 3)}
        if draw < 0.8:
            return (
                "noisy_fractional",
                {
                    "alpha": round_amount(rng.uniform(0.55, 0.9), 3),
                    "noise_width": round_amount(0.08 * max_bid),
                },
            )
        mixture = [
            {"type": "truthful", "weight": 0.30},
            {"type": "fractional", "weight": 0.40, "alpha": 0.72},
            {
                "type": "noisy_fractional",
                "weight": 0.30,
                "alpha": 0.67,
                "noise_width": round_amount(0.06 * max_bid),
            },
        ]
        return "mixed", {"mixture": mixture}

    if task_mode == "distribution_shift":
        draw = rng.random()
        if draw < 0.35:
            return "truthful", {}
        if draw < 0.75:
            return "fractional", {"alpha": round_amount(rng.uniform(0.6, 0.85), 3)}
        return (
            "noisy_fractional",
            {
                "alpha": round_amount(rng.uniform(0.6, 0.82), 3),
                "noise_width": round_amount(0.05 * max_bid),
            },
        )

    draw = rng.random()
    if draw < 0.4:
        return "fractional", {"alpha": round_amount(rng.uniform(0.55, 0.8), 3)}
    if draw < 0.7:
        return "truthful", {}
    if draw < 0.9:
        return (
            "noisy_fractional",
            {
                "alpha": round_amount(rng.uniform(0.58, 0.8), 3),
                "noise_width": round_amount(0.05 * max_bid),
            },
        )
    mixture = [
        {"type": "truthful", "weight": 0.25},
        {"type": "fractional", "weight": 0.45, "alpha": 0.70},
        {
            "type": "noisy_fractional",
            "weight": 0.30,
            "alpha": 0.66,
            "noise_width": round_amount(0.05 * max_bid),
        },
    ]
    return "mixed", {"mixture": mixture}


def describe_policy(policy_type: str, params: JsonDict) -> str:
    if policy_type == "equilibrium":
        return (
            "standard equilibrium-inspired shading for uniform private values "
            f"(approximately alpha={params['alpha']})"
        )
    if policy_type == "truthful":
        return "truthful bidding: bid equals private value"
    if policy_type == "fractional":
        return f"fractional bidding: bid = {params['alpha']} * value"
    if policy_type == "noisy_fractional":
        return (
            "noisy fractional bidding: bid = "
            f"{params['alpha']} * value + bounded noise in "
            f"[-{params['noise_width']}, {params['noise_width']}]"
        )
    mixture_parts = []
    for part in params["mixture"]:
        detail = part["type"]
        if "alpha" in part:
            detail += f"(alpha={part['alpha']})"
        mixture_parts.append(f"{detail} with weight {part['weight']}")
    return "mixed heuristic population: " + "; ".join(mixture_parts)


def compute_opponent_bid(
    value: float,
    policy_type: str,
    params: JsonDict,
    n_bidders: int,
    max_bid: float,
    rng: random.Random,
) -> float:
    if policy_type == "equilibrium":
        alpha = float(params.get("alpha", (n_bidders - 1) / n_bidders))
        return clip_value(alpha * value, 0.0, max_bid)
    if policy_type == "truthful":
        return clip_value(value, 0.0, max_bid)
    if policy_type == "fractional":
        return clip_value(float(params["alpha"]) * value, 0.0, max_bid)
    if policy_type == "noisy_fractional":
        base = float(params["alpha"]) * value
        noise = rng.uniform(-float(params["noise_width"]), float(params["noise_width"]))
        return clip_value(base + noise, 0.0, max_bid)
    if policy_type == "mixed":
        threshold = rng.random()
        cumulative = 0.0
        chosen = params["mixture"][-1]
        for component in params["mixture"]:
            cumulative += float(component["weight"])
            chosen = component
            if threshold <= cumulative:
                break
        component_type = str(chosen["type"])
        component_params = {
            key: value for key, value in chosen.items() if key not in {"type", "weight"}
        }
        return compute_opponent_bid(value, component_type, component_params, n_bidders, max_bid, rng)
    raise ValueError(f"Unsupported policy_type: {policy_type}")


def compute_difficulty_label(
    task_mode: str,
    distribution_type: str,
    n_bidders: int,
    reserve_price: float,
    opponent_policy_type: str,
    max_bid: float,
) -> str:
    score = 0
    if task_mode != "textbook":
        score += 1
    if distribution_type != "uniform":
        score += 1
    if n_bidders >= 4:
        score += 1
    if reserve_price >= 0.15 * max_bid:
        score += 1
    if opponent_policy_type in {"noisy_fractional", "mixed"}:
        score += 1
    if score <= 1:
        return "easy"
    if score == 2:
        return "medium"
    return "hard"


def compute_reference_bid(info: JsonDict) -> float | None:
    if (
        info["task_mode"] == "textbook"
        and info["distribution_type"] == "uniform"
        and info["reserve_price"] <= 0.0
        and info["opponent_policy_type"] == "equilibrium"
    ):
        return round_amount((info["n_bidders"] - 1) * info["private_value"] / info["n_bidders"], 4)
    return None


def build_prompt(info: JsonDict, require_json_output: bool) -> str:
    reserve_line = (
        "There is no reserve price."
        if info["reserve_price"] <= 0.0
        else f"The reserve price is {info['reserve_price']:.2f}; bids below the reserve cannot win."
    )
    tie_line = (
        "If the highest eligible bids tie, the winner is chosen uniformly at random among tied bidders."
        if info["tie_break_rule"] == "random"
        else "If the highest eligible bids tie, you lose the tie."
    )
    response_line = (
        "Return JSON exactly in the form "
        '{"bid": 47.5, "reasoning_summary": "optional short explanation"}. '
        'Scoring only uses "bid".'
        if require_json_output
        else "Return only one bid as a single numeric value."
    )
    return "\n".join(
        [
            "You are bidding in a sealed-bid first-price auction.",
            f"Your private value for the item is {info['private_value']:.2f}.",
            f"There are {info['n_bidders']} bidders total, including you.",
            f"Private values are drawn independently from {describe_distribution(info['distribution_type'], info['distribution_params'])}.",
            f"Opponents are modeled as using {describe_policy(info['opponent_policy_type'], info['opponent_policy_params'])}.",
            reserve_line,
            tie_line,
            (
                "Your goal is to maximize expected payoff. If you win, you pay your own bid, "
                "so payoff = private_value - bid. If you lose, payoff = 0."
            ),
            f"Valid bids are numbers between 0 and {info['max_bid']:.2f}, inclusive.",
            "Bids below 0 or above the maximum are invalid.",
            response_line,
        ]
    )


def extract_completion_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        role = completion.get("role")
        if role not in {None, "assistant"}:
            return ""
        return extract_completion_text(completion.get("content", ""))
    if isinstance(completion, list):
        pieces: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                role = item.get("role")
                if role not in {None, "assistant"}:
                    continue
                content = item.get("content", "")
                if isinstance(content, str):
                    pieces.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            pieces.append(str(block.get("text", "")))
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(completion).strip()


def coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        candidate = float(value)
        return candidate if math.isfinite(candidate) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            candidate = float(stripped)
        except ValueError:
            return None
        return candidate if math.isfinite(candidate) else None
    return None


def parse_bid(raw_text: str, max_bid: float) -> ParseResult:
    text = raw_text.strip()
    if not text:
        return ParseResult(
            bid=None,
            parse_success=False,
            json_valid=False,
            bid_in_range=False,
            parse_mode="empty",
            parse_error="empty_response",
            number_count=0,
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    else:
        if not isinstance(payload, dict):
            return ParseResult(
                bid=None,
                parse_success=False,
                json_valid=False,
                bid_in_range=False,
                parse_mode="json",
                parse_error="json_not_object",
                number_count=0,
            )
        if "bid" not in payload:
            return ParseResult(
                bid=None,
                parse_success=False,
                json_valid=False,
                bid_in_range=False,
                parse_mode="json",
                parse_error="missing_bid",
                number_count=0,
            )
        bid = coerce_numeric(payload["bid"])
        if bid is None:
            return ParseResult(
                bid=None,
                parse_success=False,
                json_valid=False,
                bid_in_range=False,
                parse_mode="json",
                parse_error="non_numeric_bid",
                number_count=0,
            )
        return ParseResult(
            bid=bid,
            parse_success=True,
            json_valid=True,
            bid_in_range=0.0 <= bid <= max_bid,
            parse_mode="json",
            parse_error="",
            number_count=1,
        )

    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return ParseResult(
            bid=None,
            parse_success=False,
            json_valid=False,
            bid_in_range=False,
            parse_mode="regex",
            parse_error="no_numeric_bid_found",
            number_count=0,
        )
    if len(numbers) > 1:
        return ParseResult(
            bid=None,
            parse_success=False,
            json_valid=False,
            bid_in_range=False,
            parse_mode="regex",
            parse_error="multiple_numbers_found",
            number_count=len(numbers),
        )

    bid = coerce_numeric(numbers[0])
    if bid is None:
        return ParseResult(
            bid=None,
            parse_success=False,
            json_valid=False,
            bid_in_range=False,
            parse_mode="regex",
            parse_error="regex_number_not_finite",
            number_count=1,
        )
    return ParseResult(
        bid=bid,
        parse_success=True,
        json_valid=False,
        bid_in_range=0.0 <= bid <= max_bid,
        parse_mode="regex",
        parse_error="",
        number_count=1,
    )


def build_scenarios(info: JsonDict, num_mc_samples: int) -> list[Scenario]:
    rng = random.Random(int(info["simulation_seed"]))
    scenarios: list[Scenario] = []
    for sample_index in range(num_mc_samples):
        opponent_bids = []
        for bidder_index in range(info["n_bidders"] - 1):
            opponent_value = sample_value(
                rng,
                info["distribution_type"],
                info["distribution_params"],
            )
            opponent_bid = compute_opponent_bid(
                value=opponent_value,
                policy_type=info["opponent_policy_type"],
                params=info["opponent_policy_params"],
                n_bidders=info["n_bidders"],
                max_bid=info["max_bid"],
                rng=rng,
            )
            opponent_bids.append(opponent_bid)
        scenarios.append({"opponent_bids": opponent_bids, "tie_draw": rng.random()})
    return scenarios


def utility_for_scenario(
    bid: float,
    private_value: float,
    reserve_price: float,
    tie_break_rule: str,
    scenario: Scenario,
) -> tuple[float, float]:
    if bid < reserve_price:
        return 0.0, 0.0

    eligible_opponent_bids = [
        float(opponent_bid)
        for opponent_bid in scenario["opponent_bids"]
        if float(opponent_bid) >= reserve_price
    ]
    if not eligible_opponent_bids:
        return private_value - bid, 1.0

    highest_opponent_bid = max(eligible_opponent_bids)
    if bid > highest_opponent_bid + EPSILON:
        return private_value - bid, 1.0
    if abs(bid - highest_opponent_bid) <= EPSILON:
        if tie_break_rule != "random":
            return 0.0, 0.0
        tied_opponents = sum(
            1 for opponent_bid in eligible_opponent_bids if abs(opponent_bid - bid) <= EPSILON
        )
        win = scenario["tie_draw"] < 1.0 / (tied_opponents + 1)
        return (private_value - bid, 1.0) if win else (0.0, 0.0)
    return 0.0, 0.0


def estimate_bid_metrics(
    bid: float,
    private_value: float,
    reserve_price: float,
    tie_break_rule: str,
    scenarios: Sequence[Scenario],
) -> tuple[float, float]:
    total_utility = 0.0
    total_wins = 0.0
    for scenario in scenarios:
        utility, win_flag = utility_for_scenario(
            bid=bid,
            private_value=private_value,
            reserve_price=reserve_price,
            tie_break_rule=tie_break_rule,
            scenario=scenario,
        )
        total_utility += utility
        total_wins += win_flag
    sample_count = max(len(scenarios), 1)
    return total_utility / sample_count, total_wins / sample_count


def candidate_bids(max_bid: float, grid_size: int, extras: Iterable[float | None] = ()) -> list[float]:
    candidates = {round(index * max_bid / (grid_size - 1), 8) for index in range(grid_size)}
    for candidate in extras:
        if candidate is not None and 0.0 <= candidate <= max_bid:
            candidates.add(round(float(candidate), 8))
    return sorted(candidates)


def search_best_response(
    private_value: float,
    reserve_price: float,
    tie_break_rule: str,
    scenarios: Sequence[Scenario],
    max_bid: float,
    grid_size: int,
    extras: Iterable[float | None] = (),
) -> tuple[float, float]:
    best_bid = 0.0
    best_utility = -math.inf
    for candidate in candidate_bids(max_bid, grid_size, extras):
        utility, win_rate_estimate = estimate_bid_metrics(
            bid=candidate,
            private_value=private_value,
            reserve_price=reserve_price,
            tie_break_rule=tie_break_rule,
            scenarios=scenarios,
        )
        if utility > best_utility + EPSILON or (
            abs(utility - best_utility) <= EPSILON and candidate < best_bid
        ):
            best_bid = candidate
            best_utility = utility
    return best_bid, best_utility


def sentinel_value(value: float | None) -> float:
    return MISSING_FLOAT if value is None else float(value)


def analyze_completion(
    completion: Any,
    info: JsonDict,
    config: AuctionEnvConfig,
) -> JsonDict:
    raw_text = extract_completion_text(completion)
    parsed = parse_bid(raw_text, max_bid=float(info["max_bid"]))
    scenarios = build_scenarios(info, config.num_mc_samples)

    reference_bid = info.get("reference_bid")
    reference_expected_utility = MISSING_FLOAT
    if reference_bid is not None:
        reference_expected_utility, reference_win_rate = estimate_bid_metrics(
            bid=float(reference_bid),
            private_value=float(info["private_value"]),
            reserve_price=float(info["reserve_price"]),
            tie_break_rule=str(info["tie_break_rule"]),
            scenarios=scenarios,
        )

    best_response_bid = MISSING_FLOAT
    best_response_expected_utility = MISSING_FLOAT
    if config.compute_best_response_baseline:
        best_response_bid, best_response_expected_utility = search_best_response(
            private_value=float(info["private_value"]),
            reserve_price=float(info["reserve_price"]),
            tie_break_rule=str(info["tie_break_rule"]),
            scenarios=scenarios,
            max_bid=float(info["max_bid"]),
            grid_size=config.best_response_grid_size,
            extras=(parsed.bid, reference_bid, float(info["private_value"])),
        )

    expected_utility = 0.0
    win_rate = 0.0
    score = float(config.invalid_bid_penalty)
    if parsed.parse_success and parsed.bid is not None and parsed.bid_in_range:
        expected_utility, win_rate = estimate_bid_metrics(
            bid=parsed.bid,
            private_value=float(info["private_value"]),
            reserve_price=float(info["reserve_price"]),
            tie_break_rule=str(info["tie_break_rule"]),
            scenarios=scenarios,
        )
        score = expected_utility
        if parsed.bid > float(info["private_value"]):
            score -= float(config.overbid_penalty)

    utility_gap_to_reference = (
        expected_utility - reference_expected_utility
        if reference_expected_utility != MISSING_FLOAT
        else MISSING_FLOAT
    )
    regret_to_best_response = (
        max(best_response_expected_utility - expected_utility, 0.0)
        if best_response_expected_utility != MISSING_FLOAT
        else MISSING_FLOAT
    )
    bid_error_to_best_response = (
        abs(parsed.bid - best_response_bid)
        if parsed.bid is not None and best_response_bid != MISSING_FLOAT
        else MISSING_FLOAT
    )

    analysis: JsonDict = {
        "score": score,
        "expected_utility": expected_utility,
        "submitted_bid": sentinel_value(parsed.bid),
        "bid_to_value_ratio": (
            parsed.bid / float(info["private_value"])
            if parsed.bid is not None and float(info["private_value"]) > 0.0
            else MISSING_FLOAT
        ),
        "win_rate_estimate": win_rate,
        "overbid_flag": 1.0 if parsed.bid is not None and parsed.bid > float(info["private_value"]) else 0.0,
        "parse_success": 1.0 if parsed.parse_success else 0.0,
        "json_valid": 1.0 if parsed.json_valid else 0.0,
        "bid_in_range": 1.0 if parsed.bid_in_range else 0.0,
        "reference_bid": sentinel_value(reference_bid),
        "reference_expected_utility": reference_expected_utility,
        "utility_gap_to_reference": utility_gap_to_reference,
        "best_response_bid": best_response_bid,
        "best_response_expected_utility": best_response_expected_utility,
        "regret_to_best_response": regret_to_best_response,
        "bid_error_to_best_response": bid_error_to_best_response,
        "normalized_score": (
            score / max(float(info["private_value"]), 1.0)
            if config.normalize_rewards
            else MISSING_FLOAT
        ),
        "parse_mode": parsed.parse_mode,
        "parse_error": parsed.parse_error,
        "number_count": float(parsed.number_count),
        "task_mode": info["task_mode"],
        "distribution_type": info["distribution_type"],
        "n_bidders": float(info["n_bidders"]),
    }
    return analysis


def metric_value(
    state: dict[str, Any] | None,
    completion: Any,
    info: JsonDict | None,
    config: AuctionEnvConfig,
    key: str,
) -> float:
    if state is None:
        return 0.0
    cached = state.get("auction_analysis_cache")
    if cached is None:
        cached = analyze_completion(completion=completion, info=info or {}, config=config)
        state["auction_analysis_cache"] = cached
        state["auction_analysis"] = cached
    value = cached.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def build_instance_record(index: int, task_mode: str, config: AuctionEnvConfig) -> JsonDict:
    rng = random.Random(config.seed + 1009 * index)
    n_bidders = int(rng.choice(config.n_bidders_options))
    distribution_type, distribution_params = sample_distribution_spec(
        task_mode=task_mode,
        rng=rng,
        max_bid=config.max_bid,
    )
    private_value = round_amount(
        sample_value(rng, distribution_type=distribution_type, params=distribution_params),
        digits=2,
    )
    reserve_price = 0.0
    if task_mode == "reserve_price":
        low, high = config.reserve_price_fraction_range
        reserve_price = round_amount(rng.uniform(low * config.max_bid, high * config.max_bid))

    opponent_policy_type, opponent_policy_params = sample_policy_spec(
        task_mode=task_mode,
        rng=rng,
        n_bidders=n_bidders,
        max_bid=config.max_bid,
    )
    instance_id = f"first_price_auction_{index:05d}"
    info: JsonDict = {
        "instance_id": instance_id,
        "task_mode": task_mode,
        "private_value": private_value,
        "n_bidders": n_bidders,
        "distribution_type": distribution_type,
        "distribution_params": distribution_params,
        "reserve_price": reserve_price,
        "tie_break_rule": config.tie_break_rule,
        "opponent_policy_type": opponent_policy_type,
        "opponent_policy_params": opponent_policy_params,
        "max_bid": round_amount(config.max_bid, 2),
        "difficulty": compute_difficulty_label(
            task_mode=task_mode,
            distribution_type=distribution_type,
            n_bidders=n_bidders,
            reserve_price=reserve_price,
            opponent_policy_type=opponent_policy_type,
            max_bid=config.max_bid,
        ),
        "simulation_seed": config.seed * 1_000_003 + index * 7_919,
    }
    info["reference_bid"] = compute_reference_bid(info)
    return {
        "instance_id": instance_id,
        "prompt": build_prompt(info, require_json_output=config.require_json_output),
        "answer": (
            ""
            if info["reference_bid"] is None
            else json.dumps({"reference_bid": info["reference_bid"]})
        ),
        "info": info,
        "task_mode": task_mode,
        "distribution_type": distribution_type,
        "n_bidders": n_bidders,
        "difficulty": info["difficulty"],
    }


def build_dataset(config: AuctionEnvConfig) -> "Dataset":
    from datasets import Dataset

    rows = [
        build_instance_record(index=index, task_mode=task_mode, config=config)
        for index, task_mode in enumerate(build_mode_schedule(config))
    ]
    return Dataset.from_list(rows)


def build_rubric(config: AuctionEnvConfig) -> "vf.Rubric":
    import verifiers as vf

    rubric = vf.Rubric()

    def score(
        completion: Any,
        info: JsonDict | None = None,
        state: dict[str, Any] | None = None,
        **unused_kwargs: Any,
    ) -> float:
        return metric_value(state=state, completion=completion, info=info, config=config, key="score")

    rubric.add_reward_func(score, weight=1.0)

    metric_names = [
        "expected_utility",
        "submitted_bid",
        "bid_to_value_ratio",
        "win_rate_estimate",
        "overbid_flag",
        "parse_success",
        "json_valid",
        "bid_in_range",
        "reference_bid",
        "reference_expected_utility",
        "utility_gap_to_reference",
        "number_count",
        "n_bidders",
    ]
    if config.compute_best_response_baseline:
        metric_names.extend(
            [
                "best_response_bid",
                "best_response_expected_utility",
                "regret_to_best_response",
                "bid_error_to_best_response",
            ]
        )
    if config.normalize_rewards:
        metric_names.append("normalized_score")

    for metric_name in metric_names:
        def metric(
            completion: Any,
            info: JsonDict | None = None,
            state: dict[str, Any] | None = None,
            *,
            metric_name_override: str = metric_name,
            **unused_kwargs: Any,
        ) -> float:
            return metric_value(
                state=state,
                completion=completion,
                info=info,
                config=config,
                key=metric_name_override,
            )

        metric.__name__ = metric_name
        rubric.add_metric(metric)

    return rubric


def load_environment(
    config: AuctionEnvConfig | None = None,
    **kwargs: Any,
) -> "vf.SingleTurnEnv":
    """Return a Prime/verifiers single-turn environment for first-price auctions.

    Environment-specific keyword arguments can either be provided via the config
    dataclass or passed directly here. Any remaining kwargs are forwarded to
    ``vf.SingleTurnEnv(...)`` unchanged.
    """

    import verifiers as vf

    env_config = config or AuctionEnvConfig()
    config_fields = {field_name for field_name in AuctionEnvConfig.__dataclass_fields__}
    overrides = {key: kwargs.pop(key) for key in list(kwargs) if key in config_fields}
    if overrides:
        env_config = replace(env_config, **overrides)

    dataset = build_dataset(env_config)
    rubric = build_rubric(env_config)
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)


__all__ = ["AuctionEnvConfig", "load_environment"]
