"""Model prices, for working out what each eval row cost."""

# Dollars per million tokens, first-party Claude API prices.
# A model missing here is refused before any paid call.
PRICES_PER_MILLION_TOKENS = {
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50},
    "claude-sonnet-5-5": {"in": 2.0, "out": 10.0},
    "claude-opus-5-5": {"in": 4.0, "out": 20.0},
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


class UnknownModelPriceError(LookupError):
    pass


def find_price(model_name: str) -> dict[str, float] | None:
    """Match by prefix, so a served name like "claude-haiku-5-5-..." still finds its price."""
    for priced_model, price in PRICES_PER_MILLION_TOKENS.items():
        if model_name.startswith(priced_model):
            return price
    return None


def calculate_cost_usd(model_name: str | None, usage: dict[str, int] | None) -> float:
    if not model_name or not usage:
        return 0.0
    price = find_price(model_name)
    if price is None:
        raise UnknownModelPriceError(f"no price for served model {model_name}")

    input_cost = usage["input_tokens"] * price["in"]
    cache_write_cost = usage.get("cache_creation_input_tokens", 0) * price["in"] * CACHE_WRITE_MULTIPLIER
    cache_read_cost = usage.get("cache_read_input_tokens", 0) * price["in"] * CACHE_READ_MULTIPLIER
    output_cost = usage["output_tokens"] * price["out"]
    return (input_cost + cache_write_cost + cache_read_cost + output_cost) / 1_000_000
