import os
from pathlib import Path
from typing import List, Tuple
import re
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = (
    PROJECT_ROOT
    / "event_datasets_final_async"
    / "swap_events_blockrcpt_list7_b57D_20250829_with_markout.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _short_addr_label(addr: str) -> str:
    if addr == "others":
        return "others"
    if isinstance(addr, str) and addr.startswith("0x") and len(addr) >= 10:
        return addr[:10]  # 0x + 8 hex chars (first 4 bytes)
    return str(addr)[:10]


def load_data() -> pl.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    df = pl.read_csv(
        INPUT_CSV,
        infer_schema_length=10000,
        schema_overrides={
            "timestamp": pl.Int64,
            "block_number": pl.Int64,
            "tx_to": pl.Utf8,
            "pool_address": pl.Utf8,
            "amount0": pl.Float64,
            "amount1": pl.Float64,
            "gas_price": pl.Float64,
            "gas_used": pl.Float64,
            "markout_usdt": pl.Float64,
            "markout_gross_usdt": pl.Float64,
            "gas_cost_usdt": pl.Float64,
            "mid_hype_usdt": pl.Float64,
            "mid_price": pl.Float64,
        },
    )

    required_cols = ["tx_to", "markout_usdt"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    return df


def summarize_by_address(df: pl.DataFrame) -> pl.DataFrame:
    # Aggregate statistics per address
    agg_exprs = [
        pl.len().alias("num_swaps"),
        pl.col("markout_usdt").sum().alias("sum_markout"),
        pl.col("markout_usdt").mean().alias("mean_markout"),
        pl.col("markout_usdt").median().alias("median_markout"),
        pl.col("markout_usdt").std(ddof=1).alias("std_markout"),
        pl.col("markout_usdt").quantile(0.05).alias("p05"),
        pl.col("markout_usdt").quantile(0.25).alias("p25"),
        pl.col("markout_usdt").quantile(0.75).alias("p75"),
        pl.col("markout_usdt").quantile(0.95).alias("p95"),
    ]
    # Optional sums for gas and gross markout
    if "gas_cost_usdt" in df.columns:
        agg_exprs.append(pl.col("gas_cost_usdt").sum().alias("sum_gas_fee_paid"))
    if "markout_gross_usdt" in df.columns:
        agg_exprs.append(
            pl.col("markout_gross_usdt").sum().alias("sum_markout_without_gas_fee")
        )
    elif "gas_cost_usdt" in df.columns:
        # fallback compute gross as net + gas
        agg_exprs.append(
            (pl.col("markout_usdt") + pl.col("gas_cost_usdt"))
            .sum()
            .alias("sum_markout_without_gas_fee")
        )

    grouped = (
        df.group_by("tx_to")
        .agg(agg_exprs)
        .with_columns(
            pl.when(
                (pl.col("sum_markout_without_gas_fee") > 0)
                & pl.col("sum_gas_fee_paid").is_not_null()
            )
            .then(pl.col("sum_gas_fee_paid") / pl.col("sum_markout_without_gas_fee"))
            .otherwise(None)
            .alias("gas_fee_share_of_gross_profit")
        )
        .sort(["num_swaps", "sum_markout"], descending=[True, True])
    )

    out_path = OUTPUT_DIR / "markout_by_address_summary.csv"
    grouped.write_csv(out_path)
    print(f"Wrote: {out_path}")
    return grouped


def plot_boxplot_top_addresses(
    df: pl.DataFrame, summary: pl.DataFrame, top_n: int = 20
) -> Path:
    top_addresses = summary.select("tx_to").head(top_n).to_series().to_list()
    # Order by num_swaps desc for the top list
    order = (
        summary.filter(pl.col("tx_to").is_in(top_addresses))
        .sort("num_swaps", descending=True)
        .select("tx_to")
        .to_series()
        .to_list()
    )

    plt.figure(figsize=(16, 8))
    data_arrays: List[List[float]] = []
    for addr in order:
        vals = (
            df.filter(pl.col("tx_to") == addr)
            .select("markout_usdt")
            .to_series()
            .to_list()
        )
        data_arrays.append(vals)
    plt.boxplot(data_arrays, tick_labels=order, showfliers=False)
    plt.xticks(rotation=75, ha="right")
    plt.title(f"Markout distribution (USDT) - Top {top_n} addresses by number of swaps")
    plt.ylabel("Markout (USDT)")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"markout_boxplot_top{top_n}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote: {out_path}")
    return out_path


def _swap_count_bins() -> List[Tuple[int, int | None, str]]:
    # (lower_inclusive, upper_inclusive_or_None_for_inf, label)
    return [
        (1, 10, "1-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, None, "1000+"),
    ]


def plot_address_counts_by_swap_bins(summary: pl.DataFrame) -> Path:
    bins = _swap_count_bins()

    def place_in_bin(n: int) -> str:
        for lower, upper, label in bins:
            if upper is None:
                if n >= lower:
                    return label
            else:
                if lower <= n <= upper:
                    return label
        return "other"

    swap_bin = summary.select(
        pl.col("num_swaps").map_elements(place_in_bin).alias("swap_bin")
    )
    tmp = summary.with_columns(swap_bin["swap_bin"]).select("swap_bin")
    counts_df = tmp.group_by("swap_bin").agg(count=pl.len())
    # Reindex to bin order
    labels = [b[2] for b in bins]
    label_to_count = {row[0]: row[1] for row in counts_df.iter_rows()}
    counts = [label_to_count.get(lbl, 0) for lbl in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.title("Number of addresses by swap-count bins")
    plt.xlabel("Number of swaps")
    plt.ylabel("Number of addresses")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "address_counts_by_swap_bins.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote: {out_path}")
    return out_path


def _safe_cols(df: pl.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def analyze_positive_avg_addresses(df: pl.DataFrame, summary: pl.DataFrame) -> Path:
    positive = summary.filter(pl.col("mean_markout") > 0).sort(
        ["num_swaps", "sum_markout"], descending=[True, True]
    )

    out_summary_path = OUTPUT_DIR / "positive_avg_markout_summary.csv"
    positive.write_csv(out_summary_path)
    print(f"Wrote: {out_summary_path}")

    base_dir = OUTPUT_DIR / "positive_addresses"
    base_dir.mkdir(parents=True, exist_ok=True)

    for row in positive.iter_rows(named=True):
        addr = row["tx_to"]
        addr_dir = base_dir / addr
        addr_dir.mkdir(parents=True, exist_ok=True)

        addr_df = df.filter(pl.col("tx_to") == addr)
        # Add derived per-swap columns for clarity
        addr_df = addr_df.with_columns(
            gas_fee_paid=pl.col("gas_cost_usdt").fill_null(0.0)
            if "gas_cost_usdt" in df.columns
            else pl.lit(None),
            markout_without_gas_fee=(
                pl.col("markout_gross_usdt")
                if "markout_gross_usdt" in df.columns
                else (pl.col("markout_usdt") + pl.col("gas_cost_usdt"))
                if "gas_cost_usdt" in df.columns
                else pl.lit(None)
            ),
        )

        cols = _safe_cols(
            addr_df,
            [
                "timestamp",
                "block_number",
                "tx_to",
                "amount0",
                "amount1",
                "markout_usdt",
                "markout_gross_usdt",
                "gas_cost_usdt",
                "gas_fee_paid",
                "markout_without_gas_fee",
            ],
        )
        addr_df.select(cols).sort(cols[0] if cols else "tx_to").write_csv(
            addr_dir / "swaps_with_markout.csv"
        )

        # Histogram of markout
        vals = addr_df.select("markout_usdt").to_series().to_list()
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=50, edgecolor="black")
        plt.title(f"Markout distribution (USDT) for {addr}")
        plt.xlabel("Markout (USDT)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(addr_dir / "markout_histogram.png", dpi=150)
        plt.close()

        # Cumulative PnL over time
        if "timestamp" in addr_df.columns:
            ts_df = (
                addr_df.sort("timestamp")
                .with_columns(cumsum_pnl=pl.col("markout_usdt").cum_sum())
                .with_columns(
                    datetime=pl.from_epoch(pl.col("timestamp"), time_unit="s")
                )
            )
            x = ts_df.select("datetime").to_series().to_numpy()
            y = ts_df.select("cumsum_pnl").to_series().to_numpy()
            plt.figure(figsize=(10, 5))
            plt.plot(x, y, linewidth=1.5, label=_short_addr_label(addr))
            plt.legend(loc="upper left", frameon=True, fontsize=8)
            plt.title(f"Cumulative Markout (USDT) over time for {addr}")
            plt.xlabel("Datetime (UTC)")
            plt.ylabel("Cumulative Markout (USDT)")
            plt.tight_layout()
            plt.savefig(addr_dir / "cumulative_markout.png", dpi=150)
            plt.close()

    return out_summary_path


def plot_cumulative_markout_positive_addresses(df: pl.DataFrame) -> Path:
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain 'timestamp' for cumulative plots")

    ts_df = (
        df.select(["tx_to", "timestamp", "markout_usdt"])
        .sort(["tx_to", "timestamp"])
        .with_columns(cumsum_pnl=pl.col("markout_usdt").cum_sum().over("tx_to"))
    )
    final_cumsum = ts_df.group_by("tx_to").agg(final=pl.col("cumsum_pnl").last())
    positive_addresses = (
        final_cumsum.filter(pl.col("final") > 0).select("tx_to").to_series().to_list()
    )

    pos_df = ts_df.filter(pl.col("tx_to").is_in(positive_addresses)).with_columns(
        datetime=pl.from_epoch(pl.col("timestamp"), time_unit="s")
    )

    plt.figure(figsize=(14, 7))
    for addr in positive_addresses:
        g = pos_df.filter(pl.col("tx_to") == addr)
        x = g.select("datetime").to_series().to_numpy()
        y = g.select("cumsum_pnl").to_series().to_numpy()
        plt.plot(x, y, linewidth=1.0, alpha=0.9, label=_short_addr_label(addr))

    n = len(positive_addresses)
    # Always show legend; place outside plot to avoid overlap
    cols = 1 if n <= 30 else 2 if n <= 60 else 3
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=6,
        ncol=cols,
        frameon=True,
    )

    plt.title("Cumulative Markout (USDT) over time for positive-PnL addresses")
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("Cumulative Markout (USDT)")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "cumulative_markout_positive_addresses.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote: {out_path}")
    return out_path


def print_trader_ranking(summary: pl.DataFrame, top_n: int | None = None) -> None:
    ranking = summary.sort("sum_markout", descending=True)
    ranking = ranking.with_row_count(name="rank").with_columns(rank=pl.col("rank") + 1)
    if top_n is not None:
        ranking = ranking.head(top_n)
    base_cols = [
        "rank",
        "tx_to",
        "num_swaps",
        "sum_markout",
        "mean_markout",
        "median_markout",
        "std_markout",
    ]
    opt_cols = []
    if "sum_gas_fee_paid" in ranking.columns:
        opt_cols.append("sum_gas_fee_paid")
    if "sum_markout_without_gas_fee" in ranking.columns:
        opt_cols.append("sum_markout_without_gas_fee")
    if "gas_fee_share_of_gross_profit" in ranking.columns:
        opt_cols.append("gas_fee_share_of_gross_profit")
    cols = base_cols + opt_cols
    rows = ranking.select(cols).to_dicts()
    print(
        tabulate(
            rows, headers="keys", tablefmt="github", floatfmt=".6f", showindex=False
        )
    )


def print_trader_ranking_by_swaps(
    summary: pl.DataFrame, top_n: int | None = None
) -> None:
    ranking = summary.sort(["num_swaps", "sum_markout"], descending=[True, True])
    ranking = ranking.with_row_count(name="rank").with_columns(rank=pl.col("rank") + 1)
    if top_n is not None:
        ranking = ranking.head(top_n)
    base_cols = [
        "rank",
        "tx_to",
        "num_swaps",
        "sum_markout",
        "mean_markout",
        "median_markout",
        "std_markout",
    ]
    opt_cols = []
    if "sum_gas_fee_paid" in ranking.columns:
        opt_cols.append("sum_gas_fee_paid")
    if "sum_markout_without_gas_fee" in ranking.columns:
        opt_cols.append("sum_markout_without_gas_fee")
    if "gas_fee_share_of_gross_profit" in ranking.columns:
        opt_cols.append("gas_fee_share_of_gross_profit")
    cols = base_cols + opt_cols
    rows = ranking.select(cols).to_dicts()
    print(
        tabulate(
            rows, headers="keys", tablefmt="github", floatfmt=".6f", showindex=False
        )
    )


def plot_market_share_over_time(df: pl.DataFrame, top_n: int = 20) -> None:
    if "timestamp" not in df.columns:
        raise ValueError(
            "Input CSV must contain 'timestamp' for market share over time plot"
        )
    ts_df = (
        df.select(["tx_to", "timestamp", "markout_usdt"])
        .sort(["tx_to", "timestamp"])
        .with_columns(cumsum_pnl=pl.col("markout_usdt").cum_sum().over("tx_to"))
    )
    # Pivot to wide
    wide = ts_df.pivot(
        values="cumsum_pnl", index="timestamp", on="tx_to", aggregate_function="last"
    ).sort("timestamp")
    # Forward-fill within each column then fill remaining nulls with 0
    data_cols = [c for c in wide.columns if c != "timestamp"]
    wide = wide.with_columns(
        [pl.col(c).fill_null(strategy="forward") for c in data_cols]
    ).with_columns([pl.col(c).fill_null(0.0) for c in data_cols])
    # Clamp negatives to zero
    wide = wide.with_columns(
        [
            pl.when(pl.col(c) < 0).then(0.0).otherwise(pl.col(c)).alias(c)
            for c in data_cols
        ]
    )
    # Compute shares
    totals = wide.select(
        pl.sum_horizontal([pl.col(c) for c in data_cols]).alias("total")
    )
    wide_shares = (
        wide.with_columns(total=totals["total"])
        .with_columns(
            [
                pl.when(pl.col("total") > 0)
                .then(pl.col(c) / pl.col("total"))
                .otherwise(0.0)
                .alias(c)
                for c in data_cols
            ]
        )
        .drop("total")
    )
    # Top N by final cumsum
    final_cumsum = {c: wide.select(pl.col(c).last()).item() for c in data_cols}
    top_cols = [
        k
        for k, _ in sorted(final_cumsum.items(), key=lambda kv: kv[1], reverse=True)[
            :top_n
        ]
    ]
    others = wide_shares.select(
        [
            "timestamp",
            pl.sum_horizontal(
                [pl.col(c) for c in data_cols if c not in top_cols]
            ).alias("others"),
        ]
    )
    shares_top = (
        wide_shares.select(["timestamp"] + top_cols)
        .join(others, on="timestamp", how="left")
        .sort("timestamp")
    )

    dt_index = (
        shares_top.select(pl.from_epoch(pl.col("timestamp"), time_unit="s"))
        .to_series()
        .to_numpy()
    )
    y = shares_top.drop("timestamp").to_numpy().T
    labels = [_short_addr_label(c) for c in shares_top.drop("timestamp").columns]

    plt.figure(figsize=(16, 9))
    plt.stackplot(dt_index, y, labels=labels)
    cols_n = 1 if len(labels) <= 30 else 2 if len(labels) <= 60 else 3
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=6,
        ncol=cols_n,
        frameon=True,
    )
    plt.title("Market share over time (by cumulative PnL)")
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("Share of cumulative PnL")
    plt.ylim(0, 1)
    plt.tight_layout()

    out_png = OUTPUT_DIR / "market_share_over_time.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote: {out_png}")

    out_csv = OUTPUT_DIR / "market_share_over_time.csv"
    shares_top.with_columns(
        datetime=pl.from_epoch(pl.col("timestamp"), time_unit="s")
    ).drop("timestamp").select(
        ["datetime"] + [c for c in shares_top.columns if c != "timestamp"]
    ).write_csv(out_csv)
    print(f"Wrote: {out_csv}")


def plot_markout_distribution_all_traders(
    df: pl.DataFrame, summary: pl.DataFrame
) -> Path:
    # Order traders by number of swaps (descending) for consistent positioning
    order = (
        summary.sort(["num_swaps", "sum_markout"], descending=[True, True])
        .select("tx_to")
        .to_series()
        .to_list()
    )
    data_by_trader: List[List[float]] = []
    for addr in order:
        data_by_trader.append(
            df.filter(pl.col("tx_to") == addr)
            .select("markout_usdt")
            .to_series()
            .to_list()
        )

    plt.figure(figsize=(max(12, len(order) * 0.05), 6))
    plt.violinplot(
        dataset=data_by_trader, showmeans=True, showextrema=False, widths=0.9
    )
    plt.title("Distribution of markout (USDT) per trader (ordered by # swaps)")
    plt.xlabel("Traders (ordered by number of swaps)")
    plt.ylabel("Markout (USDT)")
    # Hide x tick labels to avoid clutter
    plt.xticks([])
    plt.tight_layout()

    out_path = OUTPUT_DIR / "markout_distribution_all_traders.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote: {out_path}")
    return out_path


def plot_num_trades_by_trader_rank(summary: pl.DataFrame) -> Path:
    ordered = summary.sort(["num_swaps", "sum_markout"], descending=[True, True])
    counts = ordered.select("num_swaps").to_series().to_list()

    plt.figure(figsize=(max(12, len(counts) * 0.03), 6))
    plt.bar(range(1, len(counts) + 1), counts, width=0.9)
    plt.title("Number of swaps per trader (sorted by # swaps)")
    plt.xlabel("Trader rank")
    plt.ylabel("Number of swaps")
    # Skip trader names on x-axis
    plt.xticks([])
    plt.tight_layout()

    out_path = OUTPUT_DIR / "num_swaps_by_trader_rank.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote: {out_path}")
    return out_path


def _load_or_derive_mid_series_for_markout(df: pl.DataFrame) -> pl.DataFrame:
    # Try in-file columns first
    in_file_candidates = ["mid_price", "mid_hype_usdt"]
    for c in in_file_candidates:
        if c in df.columns and "timestamp" in df.columns:
            mids = (
                df.select(["timestamp", c])
                .unique(subset=["timestamp"])
                .rename({c: "mid"})
            )
            return mids

    # Fallback: try to infer date from input swaps filename and load derived mid CSV from datasets/
    # Expect pattern like ..._YYYYMMDD[_...] in INPUT_CSV name
    fname = INPUT_CSV.name
    m = re.search(r"_(\d{8})(?:_|\.)", fname)
    if not m:
        raise ValueError(
            "No mid price column found in input CSV, and could not infer date from filename to load mids."
        )
    yyyymmdd = m.group(1)
    date_str = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    mid_csv = (
        PROJECT_ROOT / "datasets" / f"hyperliquid_mid_prices_{date_str}_HYPE_USDT.csv"
    )
    if not mid_csv.exists():
        raise FileNotFoundError(
            f"Derived mid CSV not found: {mid_csv}. Generate it first with derive_hype_usdt_mid.py or place a file with 'timestamp' and mid column."
        )
    mids_df = pl.read_csv(mid_csv)
    mid_col = (
        "mid_hype_usdt"
        if "mid_hype_usdt" in mids_df.columns
        else ("mid_price" if "mid_price" in mids_df.columns else None)
    )
    if mid_col is None or "timestamp" not in mids_df.columns:
        raise ValueError(
            f"Mid CSV {mid_csv} missing required columns. Expected 'timestamp' and one of 'mid_hype_usdt' or 'mid_price'."
        )
    mids = mids_df.select(["timestamp", mid_col]).rename({mid_col: "mid"})
    return mids


def _compute_realized_variance_per_minute(mids: pl.DataFrame) -> pl.DataFrame:
    mids = (
        mids.sort("timestamp")
        .unique(subset=["timestamp"], keep="last")
        .with_columns(mid=pl.col("mid").cast(pl.Float64))
        .filter(pl.col("mid").is_not_null() & (pl.col("mid") > 0))
        .with_columns(log_mid=pl.col("mid").log())
        .with_columns(log_ret_1s=pl.col("log_mid").diff())
        .drop_nulls("log_ret_1s")
        .with_columns(minute=(pl.col("timestamp") // 60).cast(pl.Int64))
    )
    if mids.height == 0:
        return pl.DataFrame({"minute": [], "realized_variance": []})
    rv = (
        mids.group_by("minute")
        .agg(realized_variance=(pl.col("log_ret_1s") ** 2).sum())
        .sort("minute")
    )
    return rv


def volatility_markout_correlation(df: pl.DataFrame) -> None:
    # Load/derive mid price series at 1s resolution
    mids = _load_or_derive_mid_series_for_markout(df)
    rv = _compute_realized_variance_per_minute(mids)

    # Aggregate markout per minute
    swaps = df.select(["tx_to", "timestamp", "markout_usdt"]).with_columns(
        minute=(pl.col("timestamp") // 60).cast(pl.Int64)
    )
    addr_total = swaps.group_by("tx_to").agg(sum_markout=pl.col("markout_usdt").sum())
    profitable_addrs = (
        addr_total.filter(pl.col("sum_markout") > 0)
        .select("tx_to")
        .to_series()
        .to_list()
    )

    minute_pnl_prof = (
        swaps.filter(pl.col("tx_to").is_in(profitable_addrs))
        .group_by("minute")
        .agg(pnl_prof=pl.col("markout_usdt").sum())
    )
    minute_pnl_all = swaps.group_by("minute").agg(pnl_all=pl.col("markout_usdt").sum())

    df_prof = rv.join(minute_pnl_prof, on="minute", how="full").with_columns(
        realized_variance=pl.col("realized_variance").fill_null(0.0),
        pnl_prof=pl.col("pnl_prof").fill_null(0.0),
    )
    df_all = rv.join(minute_pnl_all, on="minute", how="full").with_columns(
        realized_variance=pl.col("realized_variance").fill_null(0.0),
        pnl_all=pl.col("pnl_all").fill_null(0.0),
    )

    x_prof = df_prof.select("realized_variance").to_series().to_numpy()
    y_prof = df_prof.select("pnl_prof").to_series().to_numpy()
    x_all = df_all.select("realized_variance").to_series().to_numpy()
    y_all = df_all.select("pnl_all").to_series().to_numpy()

    def _corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n = x.size
        if n < 3:
            return {
                "r": float("nan"),
                "n": n,
                "p": float("nan"),
                "std_x": float("nan"),
                "std_y": float("nan"),
            }
        r = float(np.corrcoef(x, y)[0, 1])
        # Fisher z-test for rho=0
        z = np.arctanh(max(min(r, 0.999999), -0.999999))
        se = 1.0 / np.sqrt(max(n - 3, 1))
        z_stat = z / se
        # two-sided p-value via normal CDF
        from math import erf, sqrt

        def norm_cdf(v: float) -> float:
            return 0.5 * (1.0 + erf(v / sqrt(2.0)))

        p = 2.0 * (1.0 - norm_cdf(abs(z_stat)))
        return {
            "r": r,
            "n": n,
            "p": float(p),
            "std_x": float(np.std(x, ddof=1)) if n > 1 else float("nan"),
            "std_y": float(np.std(y, ddof=1)) if n > 1 else float("nan"),
        }

    stats_prof = _corr_stats(x_prof, y_prof)
    stats_all = _corr_stats(x_all, y_all)

    # Scatter plots with correlation in title; x in RV, y in PnL (per minute)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_prof, y_prof, s=12, alpha=0.7)
    plt.title(
        f"RV vs Minute PnL (Profitable) r={stats_prof['r']:.3f}, p={stats_prof['p']:.3g}\n"
        # f"std(RV)={stats_prof['std_x']:.3g}, std(PnL)={stats_prof['std_y']:.3g}, n={stats_prof['n']}"
    )
    plt.xlabel("Realized variance per minute (sum of squared log 1s returns)")
    plt.ylabel("Sum of markout (USDT) per minute - profitable traders")
    plt.tight_layout()
    out_prof = OUTPUT_DIR / "rv_vs_pnl_profitable.png"
    plt.savefig(out_prof, dpi=150)
    plt.close()
    print(f"Wrote: {out_prof}")

    plt.figure(figsize=(10, 6))
    plt.scatter(x_all, y_all, s=12, alpha=0.7, color="tab:orange")
    plt.title(
        f"RV vs Minute PnL (All traders) r={stats_all['r']:.3f}, p={stats_all['p']:.3g}\n"
        # f"std(RV)={stats_all['std_x']:.3g}, std(PnL)={stats_all['std_y']:.3g}, n={stats_all['n']}"
    )
    plt.xlabel("Realized variance per minute (sum of squared log 1s returns)")
    plt.ylabel("Sum of markout (USDT) per minute - all traders")
    plt.tight_layout()
    out_all = OUTPUT_DIR / "rv_vs_pnl_all.png"
    plt.savefig(out_all, dpi=150)
    plt.close()
    print(f"Wrote: {out_all}")

    # Save summary stats to a text file
    summary_path = OUTPUT_DIR / "rv_vs_pnl_stats.txt"
    with open(summary_path, "w") as f:
        f.write("Profitable traders\n")
        f.write(
            f"r={stats_prof['r']:.6f}, p={stats_prof['p']:.6g}, n={stats_prof['n']}, std_rv={stats_prof['std_x']:.6g}, std_pnl={stats_prof['std_y']:.6g}\n"
        )
        f.write("All traders\n")
        f.write(
            f"r={stats_all['r']:.6f}, p={stats_all['p']:.6g}, n={stats_all['n']}, std_rv={stats_all['std_x']:.6g}, std_pnl={stats_all['std_y']:.6g}\n"
        )
    print(f"Wrote: {summary_path}")


def main() -> None:
    ensure_output_dir()
    df = load_data()
    summary = summarize_by_address(df)
    print_trader_ranking(summary)
    print_trader_ranking_by_swaps(summary)
    plot_boxplot_top_addresses(df, summary, top_n=20)
    plot_address_counts_by_swap_bins(summary)
    analyze_positive_avg_addresses(df, summary)
    plot_cumulative_markout_positive_addresses(df)
    plot_market_share_over_time(df, top_n=20)
    plot_markout_distribution_all_traders(df, summary)
    plot_num_trades_by_trader_rank(summary)
    volatility_markout_correlation(df)


if __name__ == "__main__":
    main()
