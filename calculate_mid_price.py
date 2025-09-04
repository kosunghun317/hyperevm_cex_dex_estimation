import argparse
from pathlib import Path
import polars as pl


def _infer_epoch_unit_from_max_timestamp(max_timestamp: int) -> str:
    """Infer epoch unit based on magnitude of the timestamp.

    Returns one of: "ns", "us", "ms", or "s" (we expect ms/us/s here).
    """
    if max_timestamp >= 10**18:
        return "ns"
    if max_timestamp >= 10**14:
        return "us"
    if max_timestamp >= 10**12:
        return "ms"
    if max_timestamp >= 10**9:
        return "s"
    # Extremely small/invalid epoch for modern data, default to seconds
    return "s"


def compute_mid_price_per_second(input_csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        input_csv_path,
        infer_schema_length=1000,
    )

    if "timestamp" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Input CSV must contain at least 'symbol' and 'timestamp' columns")

    # Ensure integer timestamps and infer unit (s/ms/us)
    df = df.with_columns(pl.col("timestamp").cast(pl.Int64))
    max_ts = df.select(pl.max("timestamp")).to_series().item()
    epoch_unit = _infer_epoch_unit_from_max_timestamp(int(max_ts))

    # Convert epoch to Datetime and floor to 1-second bins
    df = df.with_columns(
        ts=pl.from_epoch(pl.col("timestamp"), time_unit=epoch_unit),
        # Per-row values to support size-weighted mid within the second
        ask_amount=pl.col("ask_amount").fill_null(0.0),
        bid_amount=pl.col("bid_amount").fill_null(0.0),
        ask_price=pl.col("ask_price"),
        bid_price=pl.col("bid_price"),
    ).with_columns(
        second=pl.col("ts").dt.truncate("1s"),
        row_weight=pl.col("ask_amount") + pl.col("bid_amount"),
        row_weighted_price=pl.col("ask_price") * pl.col("ask_amount")
        + pl.col("bid_price") * pl.col("bid_amount"),
        fallback_mid=((pl.col("ask_price") + pl.col("bid_price")) / 2.0),
    )

    # Aggregate per symbol and second using size-weighted mid price
    agg = (
        df.group_by(["symbol", "second"]).agg(
            numerator=pl.col("row_weighted_price").sum(),
            denom=pl.col("row_weight").sum(),
            fallback_mid_mean=pl.col("fallback_mid").mean(),
        )
        .with_columns(
            mid_price=pl.when(pl.col("denom") > 0)
            .then(pl.col("numerator") / pl.col("denom"))
            .otherwise(pl.col("fallback_mid_mean"))
        )
        .select(["symbol", "second", "mid_price"])  # keep only needed columns
        .sort(["symbol", "second"])  # ensure order for forward-fill later
    )

    # Create a full 1s grid per symbol from min(second) to max(second)
    bounds = (
        agg.group_by("symbol")
        .agg(
            min_sec=pl.col("second").min(),
            max_sec=pl.col("second").max(),
        )
        .with_columns(
            seconds=pl.datetime_ranges(
                start=pl.col("min_sec"),
                end=pl.col("max_sec"),
                interval="1s",
                closed="both",
            )
        )
        .explode("seconds")
        .rename({"seconds": "second"})
    )

    # Left join and forward-fill missing seconds per symbol
    out = (
        bounds.join(agg, on=["symbol", "second"], how="left")
        .sort(["symbol", "second"])  # sort before ffill
        .with_columns(
            mid_price=pl.col("mid_price").fill_null(strategy="forward").over("symbol")
        )
        .with_columns(
            # Polars supports timestamp units ns/us/ms; derive seconds via ms // 1000
            timestamp=(pl.col("second").dt.timestamp(time_unit="ms") / 1000).floor().cast(pl.Int64),
        )
        .rename({"second": "datetime"})
        .select(["symbol", "datetime", "timestamp", "mid_price"])  # final schema
    )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 1-second mid price time series from quotes CSV.")
    parser.add_argument("--input", required=True, help="Path to input quotes CSV")
    parser.add_argument(
        "--output",
        required=False,
        help="Path to output CSV. Defaults to '<input_stem>_mid_1s.csv' in the same directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_mid_1s.csv")
    )

    result = compute_mid_price_per_second(input_path)
    result.write_csv(output_path)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()


