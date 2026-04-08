"""Tiny no-dependency smoke test for local development."""

from auction_env import AuctionEnvConfig, analyze_completion, build_instance_record


def main() -> None:
    config = AuctionEnvConfig(num_examples=4, num_mc_samples=24)
    row = build_instance_record(0, "textbook", config)

    assert "prompt" in row
    assert "info" in row
    assert "answer" not in row

    valid = analyze_completion('{"bid": 25}', row["info"], config)
    invalid = analyze_completion("bid 25 and 30", row["info"], config)

    assert valid["parse_success"] == 1.0
    assert valid["bid_in_range"] == 1.0
    assert invalid["parse_success"] == 0.0
    assert invalid["parse_error"] == "multiple_numbers_found"

    print("smoke_ok")


if __name__ == "__main__":
    main()
