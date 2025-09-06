import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GLOB = str(PROJECT_ROOT / "datasets" / "hyperliquid_mid_prices_*_HYPE_USDT.csv")
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _detect_mid_column(df: pd.DataFrame) -> str:
    for c in ("mid_price", "mid_hype_usdt"):
        if c in df.columns:
            return c
    raise ValueError(
        "Input mid CSV missing mid column (expected 'mid_price' or 'mid_hype_usdt')"
    )


def _load_mid_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"File {path} missing 'timestamp' column")
    mid_col = _detect_mid_column(df)
    out = (
        df[["timestamp", mid_col]]
        .rename(columns={mid_col: "mid"})
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .copy()
    )
    out["mid"] = pd.to_numeric(out["mid"], errors="coerce")
    out = out.dropna(subset=["mid"]).copy()
    out = out[out["mid"] > 0].copy()
    return out


def _compute_log_returns_1s(mids: pd.DataFrame) -> pd.DataFrame:
    mids = (
        mids.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
    )
    mids["log_mid"] = np.log(mids["mid"].astype(float))
    mids["log_ret_1s"] = mids["log_mid"].diff()
    return mids.dropna(subset=["log_ret_1s"]).copy()


def _realized_variance_nonoverlapping(mids: pd.DataFrame, window_s: int) -> pd.Series:
    if mids.empty:
        return pd.Series(dtype=float)
    with_rets = _compute_log_returns_1s(mids)
    if with_rets.empty:
        return pd.Series(dtype=float)
    base_ts = int(with_rets["timestamp"].iloc[0])
    bins = ((with_rets["timestamp"] - base_ts) // window_s).astype(int)
    rv = (
        with_rets.assign(bin=bins)
        .groupby("bin")["log_ret_1s"]
        .apply(lambda s: float((s**2).sum()))
    )
    rv.index.name = None
    return rv


def _detect_mid_column_from_file_pl(path: Path) -> str:
    # Read only schema to detect available columns
    schema_df = pl.read_csv(path, n_rows=0)
    cols = set(schema_df.columns)
    for c in ("mid_price", "mid_hype_usdt"):
        if c in cols:
            return c
    raise ValueError(
        f"Input mid CSV {path} missing mid column (expected 'mid_price' or 'mid_hype_usdt')"
    )


def _load_mid_file_pl(path: Path) -> pl.DataFrame:
    mid_col = _detect_mid_column_from_file_pl(path)
    df = pl.read_csv(path, columns=["timestamp", mid_col])
    df = (
        df.rename({mid_col: "mid"})
        .with_columns(
            pl.col("timestamp").cast(pl.Int64),
            pl.col("mid").cast(pl.Float64),
        )
        .filter(pl.col("mid").is_not_null() & (pl.col("mid") > 0))
    )
    # sort and deduplicate timestamps, keep last
    df = df.sort("timestamp").unique(subset=["timestamp"], keep="last")
    return df


def _realized_variance_nonoverlapping_pl(
    mids: pl.DataFrame, window_s: int
) -> list[float]:
    if mids.height == 0:
        return []
    with_rets = (
        mids.sort("timestamp")
        .with_columns(
            log_mid=pl.col("mid").log(),
        )
        .with_columns(log_ret_1s=pl.col("log_mid").diff())
        .drop_nulls("log_ret_1s")
    )
    if with_rets.height == 0:
        return []
    base_ts = with_rets.select(pl.first("timestamp")).to_series().item()
    binned = with_rets.with_columns(
        bin=((pl.col("timestamp") - pl.lit(base_ts)) // window_s).cast(pl.Int64)
    )
    rv = binned.group_by("bin").agg(rv=(pl.col("log_ret_1s") ** 2).sum()).sort("bin")
    return rv.select("rv").to_series().to_list()


def compute_rv_distributions(
    mid_files: List[Path], intervals_s: List[int], engine: str = "polars"
) -> List[Tuple[int, List[float]]]:
    distributions: List[Tuple[int, List[float]]] = []
    if engine == "polars":
        loaded_pl: List[pl.DataFrame] = [_load_mid_file_pl(p) for p in mid_files]
        for w in intervals_s:
            all_windows: List[float] = []
            for mids in loaded_pl:
                vals = _realized_variance_nonoverlapping_pl(mids, w)
                if vals:
                    # Convert RV over window w seconds to daily volatility: sqrt(RV * (86400 / w))
                    scale = 86400.0 / float(w)
                    daily_vol = [
                        float(np.sqrt(v * scale))
                        for v in vals
                        if np.isfinite(v) and v >= 0
                    ]
                    all_windows.extend(daily_vol)
            distributions.append((w, all_windows))
    else:
        # pandas engine
        loaded_pd: List[pd.DataFrame] = [_load_mid_file(p) for p in mid_files]
        for w in intervals_s:
            all_windows = []
            for mids in loaded_pd:
                rv = _realized_variance_nonoverlapping(mids, w)
                if not rv.empty:
                    scale = 86400.0 / float(w)
                    daily_vol = [
                        float(np.sqrt(v * scale))
                        for v in rv.tolist()
                        if np.isfinite(v) and v >= 0
                    ]
                    all_windows.extend(daily_vol)
            distributions.append((w, all_windows))
    return distributions


def plot_rv_boxplot(
    distributions: List[Tuple[int, List[float]]], out_path: Path
) -> None:
    # Filter out intervals with no data
    filtered = [(w, vals) for (w, vals) in distributions if len(vals) > 0]
    if not filtered:
        raise ValueError(
            "No realized volatility samples available to plot. Check input files and intervals."
        )

    labels = [f"{w}s" for w, _ in filtered]
    data = [vals for _, vals in filtered]

    width = max(12, len(labels) * 0.4)
    plt.figure(figsize=(width, 7))
    plt.boxplot(data, tick_labels=labels, showfliers=False, showmeans=True)
    plt.xticks(rotation=75, ha="right")
    plt.title("Daily realized volatility distributions across interval lengths")
    plt.xlabel("Interval length")
    plt.ylabel("Daily realized volatility (sqrt(RV * 86400/interval))")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute realized variance distributions across interval lengths."
    )
    parser.add_argument(
        "--mids_glob",
        default=DEFAULT_GLOB,
        help=f"Glob for input mid price CSVs (default: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        default=5,
        help="Minimum interval in seconds (inclusive)",
    )
    parser.add_argument(
        "--max_interval",
        type=int,
        default=300,
        help="Maximum interval in seconds (inclusive)",
    )
    parser.add_argument(
        "--step", type=int, default=5, help="Step in seconds between interval lengths"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "rv_boxplot_by_interval.png"),
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--engine",
        choices=["polars", "pandas"],
        default="polars",
        help="Computation engine to use",
    )
    args = parser.parse_args()

    ensure_output_dir()

    if any(ch in args.mids_glob for ch in "*?[]"):
        files = [Path(p) for p in sorted(glob.glob(args.mids_glob))]
    else:
        files = [Path(args.mids_glob)]
    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No input files matched: {args.mids_glob}")

    intervals = [
        i for i in range(args.min_interval, args.max_interval + 1, args.step) if i > 0
    ]
    distributions = compute_rv_distributions(files, intervals, engine=args.engine)
    plot_rv_boxplot(distributions, Path(args.output).resolve())


if __name__ == "__main__":
    main()
