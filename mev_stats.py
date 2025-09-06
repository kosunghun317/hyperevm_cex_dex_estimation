from pathlib import Path
import glob
from math import erf, sqrt, atanh, isfinite
from typing import List, Tuple
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
SWAPS_DIR = PROJECT_ROOT / "event_datasets_final_async"
OUTPUT_DIR = PROJECT_ROOT / "presentation/images_goal_2"

load_dotenv()


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_mid_price_files() -> List[Path]:
    pattern = str(DATASETS_DIR / "hyperliquid_mid_prices_*_HYPE_USDT.csv")
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    return [f for f in files if f.exists()]


def list_swap_event_files() -> List[Path]:
    files = [
        Path(p)
        for p in sorted(glob.glob(str(SWAPS_DIR / "swap_events_blockrcpt_*.csv")))
    ]
    return [f for f in files if f.exists()]


def _detect_mid_column_from_file_pl(path: Path) -> str:
    schema_df = pl.read_csv(path, n_rows=0)
    cols = set(schema_df.columns)
    for c in ("mid_price", "mid_hype_usdt"):
        if c in cols:
            return c
    raise ValueError(
        f"Input mid CSV {path} missing mid column (expected 'mid_price' or 'mid_hype_usdt')"
    )


def load_all_mids(mid_files: List[Path]) -> pl.DataFrame:
    frames: List[pl.DataFrame] = []
    for p in mid_files:
        mid_col = _detect_mid_column_from_file_pl(p)
        df = pl.read_csv(p, columns=["timestamp", mid_col]).rename(
            {mid_col: "mid_price"}
        )
        df = (
            df.with_columns(
                pl.col("timestamp").cast(pl.Int64),
                pl.col("mid_price").cast(pl.Float64),
            )
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
            .filter(pl.col("mid_price").is_finite() & (pl.col("mid_price") > 0))
        )
        frames.append(df)
    if not frames:
        return pl.DataFrame({"timestamp": [], "mid_price": []})
    mids = pl.concat(frames).sort("timestamp").unique(subset=["timestamp"], keep="last")
    return mids


def load_all_swaps(swap_files: List[Path]) -> pl.DataFrame:
    frames: List[pl.DataFrame] = []
    for p in swap_files:
        df = pl.read_csv(
            p,
            schema_overrides={
                "timestamp": pl.Int64,
                "block_number": pl.Int64,
                "tx_to": pl.Utf8,
                "amount0": pl.Float64,
                "amount1": pl.Float64,
                "pool_address": pl.Utf8,
                "gas_price": pl.Float64,
                "gas_used": pl.Float64,
            },
        )
        required = {
            "timestamp",
            "block_number",
            "tx_to",
            "amount0",
            "amount1",
            "pool_address",
            "gas_price",
            "gas_used",
        }
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Swap CSV {p} missing required columns: {missing}")
        frames.append(df)
    if not frames:
        return pl.DataFrame(
            {
                "timestamp": [],
                "block_number": [],
                "tx_to": [],
                "amount0": [],
                "amount1": [],
                "pool_address": [],
                "gas_price": [],
                "gas_used": [],
            }
        )
    swaps = pl.concat(frames).sort(
        ["timestamp", "block_number"]
    )  # stable enough for our use
    return swaps


def identify_searchers(
    markout_df: pl.DataFrame, address_col: str = "tx_to", pnl_col: str = "markout_usdt"
) -> pl.DataFrame:
    grouped = (
        markout_df.group_by(address_col)
        .agg(
            count=pl.len(),
            cumulative_pnl=pl.col(pnl_col).sum(),
            avg_pnl=pl.col(pnl_col).mean(),
            p25_pnl=pl.col(pnl_col).quantile(0.25, interpolation="nearest"),
            median_pnl=pl.col(pnl_col).median(),
            p75_pnl=pl.col(pnl_col).quantile(0.75, interpolation="nearest"),
            gas_consumption=pl.col("gas_used").sum(),
            # Treat non-finite as 0 before summing
            gas_fee_paid_usdt=pl.when(pl.col("gas_cost_usdt").is_finite())
            .then(pl.col("gas_cost_usdt"))
            .otherwise(0.0)
            .sum(),
            pnl_without_gas_fee=pl.when(pl.col("markout_gross_usdt").is_finite())
            .then(pl.col("markout_gross_usdt"))
            .otherwise(0.0)
            .sum(),
        )
        .with_columns(
            gas_payment_ratio=pl.when(pl.col("pnl_without_gas_fee") != 0)
            .then(pl.col("gas_fee_paid_usdt") / pl.col("pnl_without_gas_fee"))
            .otherwise(None)
        )
        .filter((pl.col("count") > 30) & (pl.col("cumulative_pnl") > 0))
        .sort("cumulative_pnl", descending=True)
    )
    return grouped


def compute_market_share_over_time(
    markout_df: pl.DataFrame,
    searchers: List[str],
    address_col: str = "tx_to",
    pnl_col: str = "markout_usdt",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df = markout_df.filter(pl.col(address_col).is_in(searchers)).with_columns(
        pl.col("timestamp").cast(pl.Int64)
    )
    df_sec = (
        df.group_by(["timestamp", address_col])
        .agg(**{pnl_col: pl.col(pnl_col).sum()})
        .sort([address_col, "timestamp"])
        .with_columns(cum_pnl=pl.col(pnl_col).cum_sum().over(address_col))
    )

    # Pivot to wide cum_pnl by timestamp
    wide = df_sec.pivot(index="timestamp", on=address_col, values="cum_pnl").sort(
        "timestamp"
    )
    # Forward-fill per column; replace nulls with 0
    wide_ffill = wide.select(
        [pl.col("timestamp")]
        + [
            pl.col(c).forward_fill().fill_null(0.0).alias(c)
            for c in wide.columns
            if c != "timestamp"
        ]
    )
    # Positive-only cumulative PnL to avoid negative shares; re-normalize so rows sum to 1
    searcher_cols = [c for c in wide_ffill.columns if c != "timestamp"]
    positive_cum = wide_ffill.select(
        ["timestamp"]
        + [pl.col(c).clip(lower_bound=0.0).alias(c) for c in searcher_cols]
    )
    total_pos = positive_cum.select(
        ["timestamp", pl.sum_horizontal(pl.col(searcher_cols)).alias("total_pos_pnl")]
    )

    market_share = (
        positive_cum.join(total_pos, on="timestamp", how="left")
        .select(
            ["timestamp"]
            + [
                (
                    pl.when(pl.col("total_pos_pnl") > 0.0)
                    .then(pl.col(c) / pl.col("total_pos_pnl"))
                    .otherwise(0.0)
                ).alias(c)
                for c in searcher_cols
            ]
        )
        .sort("timestamp")
    )

    long = (
        positive_cum.unpivot(
            index=["timestamp"],
            on=searcher_cols,
            variable_name="searcher",
            value_name="cum_pnl_pos",
        )
        .join(total_pos, on="timestamp", how="left")
        .with_columns(
            market_share=pl.when(pl.col("total_pos_pnl") > 0.0)
            .then(pl.col("cum_pnl_pos") / pl.col("total_pos_pnl"))
            .otherwise(0.0)
        )
    )

    return market_share, long


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def correlation_p_value(r: float, n: int) -> float:
    # Fisher z-transform approximation for p-value (no SciPy dependency)
    r = max(min(r, 0.999999), -0.999999)
    if n <= 3:
        return float("nan")
    z = atanh(r) * sqrt(max(n - 3, 1))
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    return p


def compute_minute_realized_variance(mids: pl.DataFrame) -> pl.DataFrame:
    if mids.height == 0:
        return pl.DataFrame({"minute": [], "rv_minute": []})
    with_rets = (
        mids.sort("timestamp")
        .with_columns(log_mid=pl.col("mid_price").log())
        .with_columns(log_ret_1s=pl.col("log_mid").diff())
        .drop_nulls("log_ret_1s")
        .with_columns(minute=(pl.col("timestamp") // 60).cast(pl.Int64))
    )
    if with_rets.height == 0:
        return pl.DataFrame({"minute": [], "rv_minute": []})
    rv_min = (
        with_rets.group_by("minute")
        .agg(rv_minute=(pl.col("log_ret_1s") ** 2).sum())
        .sort("minute")
    )
    return rv_min


def plot_market_share(market_share: pl.DataFrame, out_path: Path) -> None:
    if market_share.height == 0:
        return
    cols = [c for c in market_share.columns if c != "timestamp"]
    if not cols:
        return
    # The computed shares are already non-negative and row-normalized to 1
    pdf = market_share.select(["timestamp"] + cols).to_pandas()
    pdf["datetime"] = pd.to_datetime(pdf["timestamp"], unit="s", utc=True)
    pdf = pdf.set_index("datetime")
    plt.figure(figsize=(14, 6))
    (pdf[cols] * 100.0).plot(kind="area", stacked=True, figsize=(14, 6), alpha=0.9)
    plt.ylabel("Market share (%)")
    plt.xlabel("Time (UTC)")
    plt.title("Searchers' cumulative PnL market share over time (top by PnL)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_rv_vs_pnl_scatter(
    rv_minute: pl.DataFrame, pnl_minute: pl.DataFrame, out_path: Path, stats_out: Path
) -> None:
    merged = pnl_minute.join(rv_minute, on="minute", how="inner")
    if merged.height == 0:
        return

    # Remove a known bad minute and trim extreme outliers (top/bottom 0.1% by RV)
    merged = merged.filter(pl.col("minute") != 29269385)
    if merged.height == 0:
        return
    q_df = merged.select(
        q_low=pl.col("rv_minute").quantile(0.001, interpolation="linear"),
        q_high=pl.col("rv_minute").quantile(0.999, interpolation="linear"),
    )
    # q_df = merged
    try:
        q_low = float(q_df.get_column("q_low").item())
        q_high = float(q_df.get_column("q_high").item())
        if np.isfinite(q_low) and np.isfinite(q_high) and q_high >= q_low:
            merged = merged.filter(
                (pl.col("rv_minute") >= q_low) & (pl.col("rv_minute") <= q_high)
            )
    except Exception:
        pass
    if merged.height == 0:
        return
    np_df = merged.select(["minute", "rv_minute", "pnl_minute"]).to_numpy()
    t = np.asarray(np_df[:, 0], dtype=int)
    rv_min = np.asarray(np_df[:, 1], dtype=float)
    y = np.asarray(np_df[:, 2], dtype=float)
    # Convert minute RV to dailyized volatility: sqrt(RV * (86400/60))
    daily_scale = 86400.0 / 60.0
    x = np.maximum(rv_min, 0.0) * daily_scale
    # Fit linear regression y = a + b x (via polyfit)
    if len(x) >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        b, a = np.polyfit(x, y, 1)  # returns slope, intercept with deg=1
        r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
        p = correlation_p_value(float(r), len(x)) if isfinite(r) else float("nan")
    else:
        a, b, r, p = float("nan"), float("nan"), float("nan"), float("nan")

    # Convert minute to datetime for x-axis if desired (but x-axis is RV here)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=12, alpha=0.6)
    if isfinite(b) and isfinite(a):
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ys = a + b * xs
        plt.plot(xs, ys, color="red", linewidth=2, label="OLS fit")
        plt.legend()
    plt.xlabel("Dailyized realized variance (RV_min * 86400/60)")
    plt.ylabel("Sum of searchers' PnL per minute (USDT)")
    title = f"RV vs Searcher PnL per minute\nbeta={b:.4g}, p={p:.12g}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Write stats and the merged data for inspection
    stats_text = (
        f"n={len(x)}\n"
        f"slope_beta={b}\n"
        f"intercept_alpha={a}\n"
        f"pearson_r={r}\n"
        f"p_value_approx={p}\n"
    )
    stats_out.write_text(stats_text)


def compute_markout_pl(
    swaps: pl.DataFrame,
    mids: pl.DataFrame,
    hype_decimals: int = 18,
    usdt_decimals: int = 6,
    markout_column_name: str = "markout_usdt",
) -> pl.DataFrame:
    hype_scale = float(10**hype_decimals)
    usdt_scale = float(10**usdt_decimals)

    swaps2 = swaps.with_columns(
        amount0_adj=pl.col("amount0").cast(pl.Float64) / hype_scale,
        amount1_adj=pl.col("amount1").cast(pl.Float64) / usdt_scale,
    ).with_columns(
        amount0_trader_adj=(-pl.col("amount0_adj")),
        amount1_trader_adj=(-pl.col("amount1_adj")),
        t_plus_1=(pl.col("timestamp").cast(pl.Int64) + 1).cast(pl.Int64),
    )

    mids2 = mids.rename({"timestamp": "t_plus_1", "mid_price": "mid_t_plus_1"})

    merged = swaps2.join(mids2, on="t_plus_1", how="inner")

    # Sanitize gas inputs to ensure gas cost is always well-defined (non-null, non-negative)
    merged = merged.with_columns(
        gas_price_sanitized=pl.col("gas_price")
        .cast(pl.Float64)
        .fill_null(0.0)
        .clip(lower_bound=0.0),
        gas_used_sanitized=pl.col("gas_used")
        .cast(pl.Float64)
        .fill_null(0.0)
        .clip(lower_bound=0.0),
    ).with_columns(
        markout_gross_usdt=pl.col("mid_t_plus_1") * pl.col("amount0_trader_adj")
        + pl.col("amount1_trader_adj"),
        gas_cost_usdt=pl.col("gas_price_sanitized")
        * pl.col("gas_used_sanitized")
        * pl.col("mid_t_plus_1")
        / hype_scale,
    )

    merged = merged.with_columns(
        **{markout_column_name: pl.col("markout_gross_usdt") - pl.col("gas_cost_usdt")}
    ).drop(["t_plus_1"])  # keep *_adj for inspection

    return merged


def compute_rv_distributions_pl(
    mid_files: List[Path], intervals_s: List[int]
) -> List[Tuple[int, List[float]]]:
    def _load_mid_file_pl(path: Path) -> pl.DataFrame:
        mid_col = _detect_mid_column_from_file_pl(path)
        df = (
            pl.read_csv(path, columns=["timestamp", mid_col])
            .rename({mid_col: "mid"})
            .with_columns(
                pl.col("timestamp").cast(pl.Int64),
                pl.col("mid").cast(pl.Float64),
            )
            .filter(pl.col("mid").is_not_null() & (pl.col("mid") > 0))
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
        )
        return df

    def _realized_variance_nonoverlapping_pl(
        mids: pl.DataFrame, window_s: int
    ) -> list[float]:
        if mids.height == 0:
            return []
        with_rets = (
            mids.sort("timestamp")
            .with_columns(log_mid=pl.col("mid").log())
            .with_columns(log_ret_1s=pl.col("log_mid").diff())
            .drop_nulls("log_ret_1s")
        )
        if with_rets.height == 0:
            return []
        base_ts = with_rets.select(pl.first("timestamp")).to_series().item()
        binned = with_rets.with_columns(
            bin=((pl.col("timestamp") - pl.lit(base_ts)) // window_s).cast(pl.Int64)
        )
        rv = (
            binned.group_by("bin").agg(rv=(pl.col("log_ret_1s") ** 2).sum()).sort("bin")
        )
        return rv.select("rv").to_series().to_list()

    loaded = [_load_mid_file_pl(p) for p in mid_files]
    distributions: List[Tuple[int, List[float]]] = []
    for w in intervals_s:
        all_windows: List[float] = []
        for mids in loaded:
            vals = _realized_variance_nonoverlapping_pl(mids, w)
            if vals:
                scale = 86400.0 / float(w)
                daily_vol = [
                    float(np.sqrt(v * scale)) for v in vals if np.isfinite(v) and v >= 0
                ]
                all_windows.extend(daily_vol)
        distributions.append((w, all_windows))
    return distributions


def plot_rv_boxplot_pl(
    distributions: List[Tuple[int, List[float]]], out_path: Path
) -> None:
    filtered = [(w, vals) for (w, vals) in distributions if len(vals) > 0]
    if not filtered:
        return
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


def _short_addr(addr: str, head: int = 6, tail: int = 4) -> str:
    if not isinstance(addr, str):
        return str(addr)
    if len(addr) <= head + tail + 2:
        return addr
    return addr[: head + 2] + "…" + addr[-tail:]


def plot_searcher_panels(
    markout: pl.DataFrame, searcher_summary: pl.DataFrame, out_path: Path
) -> None:
    if searcher_summary.height == 0:
        return
    searchers = searcher_summary.select("searcher").to_series().to_list()
    # Collect per-searcher pnl arrays in the same order as summary
    pnl_series_per_searcher: list[list[float]] = []
    for s in searchers:
        arr = (
            markout.filter(pl.col("tx_to") == s)
            .select("markout_usdt")
            .to_series()
            .to_list()
        )
        pnl_series_per_searcher.append([float(x) for x in arr if x is not None])

    # Scalars from summary in same order
    cum_pnl = searcher_summary.select("cumulative_pnl").to_series().to_list()
    counts = searcher_summary.select("count").to_series().to_list()
    gas_ratio = searcher_summary.select("gas_payment_ratio").to_series().to_list()

    labels = [_short_addr(s) for s in searchers]

    # Create 2x2 panel
    n = len(searchers)
    width = max(14, min(30, 2 + 0.6 * n))
    fig, axes = plt.subplots(2, 2, figsize=(width, 10))

    # 1) Box plot of PnL distribution per searcher
    ax = axes[0, 0]
    bp = ax.boxplot(
        pnl_series_per_searcher,
        showfliers=False,
        tick_labels=labels,
        vert=True,
        patch_artist=True,
    )
    ax.set_title("PnL distribution by searcher")
    ax.set_ylabel("PnL per swap (USDT)")
    ax.tick_params(axis="x", rotation=75)

    # 2) Cumulative PnL per searcher (bar)
    ax = axes[0, 1]
    ax.bar(range(n), cum_pnl)
    ax.set_title("Cumulative PnL by searcher")
    ax.set_ylabel("Cumulative PnL (USDT)")
    ax.set_xticks(range(n), labels, rotation=75, ha="right")

    # 3) Number of swaps per searcher (bar)
    ax = axes[1, 0]
    ax.bar(range(n), counts)
    ax.set_title("Number of swaps by searcher")
    ax.set_ylabel("Count of swaps")
    ax.set_xticks(range(n), labels, rotation=75, ha="right")

    # 4) Gas fee ratio per searcher (bar)
    ax = axes[1, 1]
    # Replace None with np.nan for plotting
    gas_ratio_plot = [float(x) if x is not None else np.nan for x in gas_ratio]
    ax.bar(range(n), gas_ratio_plot)
    ax.set_title("Gas payment ratio (gas / gross PnL)")
    ax.set_ylabel("Ratio")
    ax.set_xticks(range(n), labels, rotation=75, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    # Using hardcoded directories relative to script location
    # DATASETS_DIR, SWAPS_DIR, OUTPUT_DIR are already set at module scope
    ensure_output_dir()

    # Load inputs
    mid_files = list_mid_price_files()
    if not mid_files:
        raise FileNotFoundError(f"No mid price files found in {DATASETS_DIR}")
    swap_files = list_swap_event_files()
    if not swap_files:
        raise FileNotFoundError(f"No swap event files found in {SWAPS_DIR}")

    print(f"Loading {len(mid_files)} mid price files…")
    mids = load_all_mids(mid_files)
    if mids.height == 0:
        raise RuntimeError("No mid data loaded")
    mid_ts_min = int(mids.select(pl.min("timestamp")).to_series().item())
    mid_ts_max = int(mids.select(pl.max("timestamp")).to_series().item())
    print(f"Loaded mids: {mids.height:,} rows, {mid_ts_min} – {mid_ts_max}")

    print(f"Loading {len(swap_files)} swap event files…")
    swaps = load_all_swaps(swap_files)
    print(f"Loaded swaps: {swaps.height:,} rows")

    # Do NOT filter swaps here to avoid dropping late-month tails prematurely.
    # We will instead compute markout then drop rows lacking coverage.

    # Compute t+1s markout with gas cost
    print("Computing 1s markout (net of gas)…")
    markout = compute_markout_pl(
        swaps,
        mids,
        hype_decimals=18,
        usdt_decimals=6,
        markout_column_name="markout_usdt",
    )
    # After markout, drop rows with undefined or non-finite values (e.g., missing mids at t+1 or invalid gas)
    markout = markout.filter(
        pl.all_horizontal(
            pl.col("markout_usdt").is_not_null() & pl.col("markout_usdt").is_finite(),
            pl.col("markout_gross_usdt").is_not_null()
            & pl.col("markout_gross_usdt").is_finite(),
            pl.col("gas_cost_usdt").is_not_null() & pl.col("gas_cost_usdt").is_finite(),
        )
    )
    print(f"Markout rows (matched at t+1): {markout.height:,}")

    # Identify searchers (tx_to) with > 30 tx and positive cumulative pnl
    print("Identifying searchers…")
    searcher_summary = (
        identify_searchers(markout)
        .rename({"tx_to": "searcher"})
        .with_columns(
            # replace any remaining NaNs with 0 for summary robustness
            pl.all().fill_null(0)
        )
    )
    searcher_summary.write_csv(OUTPUT_DIR / "markout_by_address_summary.csv")
    print(f"Wrote searcher summary: {OUTPUT_DIR / 'markout_by_address_summary.csv'}")

    if searcher_summary.height == 0:
        print(
            "No searchers found (criteria: >30 tx and positive cumulative PnL). Exiting."
        )
        return

    # Market share over time (based on cumulative PnL)
    print("Computing market share over time…")
    searcher_list = searcher_summary.select("searcher").to_series().to_list()
    market_share, market_share_long = compute_market_share_over_time(
        markout, searcher_list
    )
    market_share.write_csv(OUTPUT_DIR / "market_share_over_time.csv")
    market_share_long.write_csv(OUTPUT_DIR / "market_share_over_time_long.csv")
    print(
        f"Wrote market share time series: {OUTPUT_DIR / 'market_share_over_time.csv'} and long format"
    )
    # Plot market share area chart for top searchers
    plot_market_share(market_share, OUTPUT_DIR / "market_share_over_time.png")
    print(f"Wrote market share plot: {OUTPUT_DIR / 'market_share_over_time.png'}")

    # Realized variance per minute and relation with searchers' PnL per minute
    print("Computing realized variance per minute…")
    rv_minute = compute_minute_realized_variance(mids)
    # Searcher PnL per minute (sum over all searchers only)
    searcher_set = set(searcher_summary.select("searcher").to_series().to_list())
    markout_searchers = markout.filter(
        pl.col("tx_to").is_in(list(searcher_set))
    ).with_columns(minute=(pl.col("timestamp").cast(pl.Int64) // 60).cast(pl.Int64))
    pnl_minute = (
        markout_searchers.group_by("minute")
        .agg(pnl_minute=pl.col("markout_usdt").sum())
        .sort("minute")
    )
    pnl_minute.write_csv(OUTPUT_DIR / "searcher_pnl_per_minute.csv")
    rv_minute.write_csv(OUTPUT_DIR / "realized_variance_per_minute.csv")

    # Scatter plot with regression stats in title
    plot_rv_vs_pnl_scatter(
        rv_minute=rv_minute,
        pnl_minute=pnl_minute,
        out_path=OUTPUT_DIR / "rv_vs_searcher_pnl_per_minute.png",
        stats_out=OUTPUT_DIR / "rv_vs_searcher_pnl_stats.txt",
    )
    print(f"Wrote RV vs PnL scatter and stats under {OUTPUT_DIR}")

    # Realized variance distributions across intervals (5s to 300s)
    print("Computing realized variance distributions across intervals (5s–300s)…")
    intervals = list(range(5, 301, 5))
    distributions = compute_rv_distributions_pl(mid_files, intervals)
    plot_rv_boxplot_pl(distributions, OUTPUT_DIR / "rv_boxplot_by_interval.png")

    # Searcher panels (4-in-1)
    plot_searcher_panels(
        markout=markout,
        searcher_summary=searcher_summary,
        out_path=OUTPUT_DIR / "searcher_panels.png",
    )

    print("Done. Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
