import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
import gzip
import shutil
from dotenv import load_dotenv

from calculate_mid_price import compute_mid_price_per_second


def _daterange_inclusive(
    start_date: datetime, end_date: datetime
) -> Iterable[datetime]:
    current = start_date
    while current <= end_date:
        yield current
        current = current + timedelta(days=1)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_quotes_for_day(
    date: datetime,
    datasets_dir: Path,
    symbols: list[str],
    api_key: Optional[str],
) -> None:
    # Lazy import to avoid dependency unless downloading
    from tardis_dev import datasets

    iso_from = date.strftime("%Y-%m-%dT00:00:00.000Z")
    iso_to = date.strftime("%Y-%m-%dT23:59:59.999Z")

    # Tardis saves to CWD by default; use datasets_dir as cwd
    cwd = os.getcwd()
    try:
        os.chdir(datasets_dir)
        datasets.download(
            exchange="hyperliquid",
            data_types=["quotes"],
            from_date=iso_from,
            to_date=iso_to,
            symbols=symbols,
            api_key=api_key,
        )
    finally:
        os.chdir(cwd)


def _build_expected_files(date: datetime, datasets_dir: Path) -> dict[str, Path]:
    day = date.strftime("%Y-%m-%d")
    return {
        "@107": datasets_dir / f"hyperliquid_quotes_{day}_@107.csv.gz",
        "@166": datasets_dir / f"hyperliquid_quotes_{day}_@166.csv.gz",
    }


def _gunzip_to_csv(gz_path: Path) -> Path:
    if not gz_path.exists():
        raise FileNotFoundError(f"Gzip file not found: {gz_path}")
    # Remove only the .gz suffix, keeping .csv
    csv_path = gz_path.with_suffix("")
    if csv_path.exists():
        return csv_path
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return csv_path


def _compute_mids_for_day(files_gz: dict[str, Path]) -> pl.DataFrame:
    # Ensure uncompressed CSVs exist
    csv_107 = _gunzip_to_csv(files_gz["@107"])
    csv_166 = _gunzip_to_csv(files_gz["@166"])

    df_107 = compute_mid_price_per_second(
        csv_107
    )  # columns: symbol, datetime, timestamp, mid_price
    df_166 = compute_mid_price_per_second(
        csv_166
    )  # columns: symbol, datetime, timestamp, mid_price

    df_107 = (
        df_107.rename({"mid_price": "mid_hype_usdc"})
        .select(["timestamp", "mid_hype_usdc"])
        .unique(maintain_order=True)
    )
    df_166 = (
        df_166.rename({"mid_price": "mid_usdt_usdc"})
        .select(["timestamp", "mid_usdt_usdc"])
        .unique(maintain_order=True)
    )

    joined = df_107.join(df_166, on="timestamp", how="inner").with_columns(
        mid_hype_usdt=pl.col("mid_hype_usdc") / pl.col("mid_usdt_usdc")
    )

    return joined.select(
        ["timestamp", "mid_hype_usdc", "mid_usdt_usdc", "mid_hype_usdt"]
    ).sort("timestamp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hyperliquid quotes and derive HYPE-USDT mid prices per day."
    )
    parser.add_argument("--from_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--to_date", required=True, help="End date (YYYY-MM-DD), inclusive"
    )
    parser.add_argument(
        "--datasets_dir",
        required=False,
        default=str(Path(__file__).resolve().parent / "datasets"),
        help="Directory to store and read quotes CSV.GZ files.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default=str(Path(__file__).resolve().parent / "datasets"),
        help="Directory to write derived mid price CSV files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="If set, download quotes for each day before processing.",
    )

    args = parser.parse_args()

    load_dotenv()
    tardis_api_key = os.getenv("TARDIS_API_KEY")

    datasets_dir = Path(args.datasets_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_dir(datasets_dir)
    _ensure_dir(output_dir)

    start = datetime.strptime(args.from_date, "%Y-%m-%d")
    end = datetime.strptime(args.to_date, "%Y-%m-%d")

    symbols = ["@107", "@166"]

    for day in _daterange_inclusive(start, end):
        expected = _build_expected_files(day, datasets_dir)

        if args.download:
            _download_quotes_for_day(day, datasets_dir, symbols, tardis_api_key)

        for sym, file_path in expected.items():
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing quotes file for {sym} on {day.date()}: {file_path}"
                )

        mids = _compute_mids_for_day(expected)

        out_path = (
            output_dir
            / f"hyperliquid_mid_prices_{day.strftime('%Y-%m-%d')}_HYPE_USDT.csv"
        )
        mids.write_csv(out_path)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
