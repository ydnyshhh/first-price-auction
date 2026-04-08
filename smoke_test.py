"""Tiny no-dependency smoke test for local development."""

from auction_env import AuctionEnvConfig, analyze_completion, build_instance_record


def main() -> None:
    config = AuctionEnvConfig(num_examples=4, num_mc_samples=24)
    row = build_instance_record(0, "textbook", config)

    assert "prompt" in row
    assert "info" in row
    assert "answer" not in row

    valid_once = analyze_completion('{"bid": 25}', row["info"], config)
    valid_twice = analyze_completion('{"bid": 25}', row["info"], config)
    invalid_parse = analyze_completion("bid 25 and 30", row["info"], config)
    negative_bid = analyze_completion('{"bid": -5}', row["info"], config)
    too_large_bid = analyze_completion('{"bid": 1000}', row["info"], config)

    assert valid_once["parse_success"] == 1.0
    assert valid_once["bid_in_range"] == 1.0
    assert valid_once["json_valid"] == 1.0

    assert valid_once["score"] == valid_twice["score"]
    assert valid_once["expected_utility"] == valid_twice["expected_utility"]
    assert valid_once["best_response_bid"] == valid_twice["best_response_bid"]
    assert valid_once["best_response_expected_utility"] == valid_twice["best_response_expected_utility"]

    assert invalid_parse["parse_success"] == 0.0
    assert invalid_parse["parse_error"] == "multiple_numbers_found"

    assert negative_bid["parse_success"] == 1.0
    assert negative_bid["bid_in_range"] == 0.0
    assert negative_bid["score"] == config.invalid_bid_penalty

    assert too_large_bid["parse_success"] == 1.0
    assert too_large_bid["bid_in_range"] == 0.0
    assert too_large_bid["score"] == config.invalid_bid_penalty

    print("smoke_ok")


if __name__ == "__main__":
    main()
