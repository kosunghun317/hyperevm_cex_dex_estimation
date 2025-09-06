import argparse
import os
from typing import Optional

import pandas as pd


def load_swaps(csv_path: str) -> pd.DataFrame:
    """Load swap events CSV and ensure required columns exist.

    Expected columns: `timestamp,block_number,tx_to,amount0,amount1,pool_address,gas_price,gas_used`
    """
    df = pd.read_csv(csv_path)
    required_cols = [
        "timestamp",
        "block_number",
        "tx_to",
        "amount0",
        "amount1",
        "pool_address",
        "gas_price",
        "gas_used",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Swap CSV missing required columns: {missing}")
    return df


def load_mid_prices(csv_path: str, mid_price_column: str | None = None) -> pd.DataFrame:
    """Load mid prices CSV and return `timestamp` and a column named `mid_price`.

    Auto-detects the mid price column if not provided. Supported names include
    `mid_price` and `mid_hype_usdt` (from derived HYPE-USDT mids).
    """
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("Mid-price CSV missing required column: 'timestamp'")

    candidate_cols = []
    if mid_price_column:
        candidate_cols.append(mid_price_column)
    # Common defaults / fallbacks
    candidate_cols.extend(["mid_price", "mid_hype_usdt"])

    chosen: str | None = None
    for c in candidate_cols:
        if c in df.columns:
            chosen = c
            break
    if chosen is None:
        raise ValueError(
            "Could not find a mid-price column. Tried: " + ", ".join(candidate_cols)
        )

    out = df[["timestamp", chosen]].rename(columns={chosen: "mid_price"})
    return out


def compute_markout(
    swaps: pd.DataFrame,
    mids: pd.DataFrame,
    hype_decimals: int = 18,
    usdt_decimals: int = 6,
    markout_column_name: str = "markout_usdt",
) -> pd.DataFrame:
    """Compute t+1 mid-price markout for each swap.

    - token0 (amount0) is HYPE with 18 decimals
    - token1 (amount1) is USDT with 6 decimals
    - Use mid price at t+1 second when swap occurs at t
    - Note: `amount0` and `amount1` are pool balance deltas. Trader deltas are the negatives.
    - Gross markout (USDT): mid_{t+1} * (-amount0_adj) + (-amount1_adj)
    - Gas fee (USDT): gas_price * gas_used * mid_{t+1} / 10**hype_decimals
    - Net markout (USDT): gross - gas_fee_usdt (stored in `markout_column_name`)

    Notes:
    - Raw amounts are adjusted by dividing by 10**decimals.
    - Sign convention in the swap file is preserved.
    """
    swaps = swaps.copy()
    mids = mids.copy()

    # Scale raw on-chain amounts by token decimals
    hype_scale = 10**hype_decimals
    usdt_scale = 10**usdt_decimals
    swaps["amount0_adj"] = swaps["amount0"].astype(float) / hype_scale
    swaps["amount1_adj"] = swaps["amount1"].astype(float) / usdt_scale
    # Convert pool deltas to trader deltas (negate)
    swaps["amount0_trader_adj"] = -swaps["amount0_adj"]
    swaps["amount1_trader_adj"] = -swaps["amount1_adj"]

    # Align mid at t+1 by joining on timestamp+1
    swaps["t_plus_1"] = swaps["timestamp"].astype(int) + 1
    mids_renamed = mids.rename(
        columns={"timestamp": "t_plus_1", "mid_price": "mid_t_plus_1"}
    )

    # Merge using inner join to intersect available timestamps
    merged = swaps.merge(mids_renamed, on="t_plus_1", how="inner")

    # Compute gross markout in USDT terms using trader deltas
    merged["markout_gross_usdt"] = (
        merged["mid_t_plus_1"] * merged["amount0_trader_adj"]
        + merged["amount1_trader_adj"]
    )

    # Compute gas cost in USDT using t+1 mid price
    hype_scale = 10**hype_decimals
    merged["gas_cost_usdt"] = (
        merged["gas_price"].astype(float)
        * merged["gas_used"].astype(float)
        * merged["mid_t_plus_1"]
        / hype_scale
    )

    # Net markout (gross - gas)
    merged[markout_column_name] = merged["markout_gross_usdt"] - merged["gas_cost_usdt"]

    # Drop helper columns
    merged = merged.drop(columns=["t_plus_1"])  # keep amount*_adj for inspection

    return merged


def main(
    swaps_csv: str,
    mid_prices_csv: str,
    output_csv: Optional[str] = None,
    hype_decimals: int = 18,
    usdt_decimals: int = 6,
    mid_price_column: Optional[str] = None,
) -> str:
    swaps = load_swaps(swaps_csv)
    mids = load_mid_prices(mid_prices_csv, mid_price_column=mid_price_column)
    result = compute_markout(
        swaps, mids, hype_decimals=hype_decimals, usdt_decimals=usdt_decimals
    )

    if output_csv is None:
        base, ext = os.path.splitext(os.path.basename(swaps_csv))
        output_csv = os.path.join(
            os.path.dirname(swaps_csv), f"{base}_with_markout{ext}"
        )

    result.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute t+1 mid-price markout for swap events."
    )
    parser.add_argument("--swaps_csv", required=True, help="Path to swap events CSV")
    parser.add_argument(
        "--mid_prices_csv", required=True, help="Path to mid prices CSV"
    )
    parser.add_argument(
        "--output_csv",
        required=False,
        default=None,
        help="Optional output CSV path; defaults to <swaps>_with_markout.csv",
    )
    # Decimals hardcoded by default per instruction
    parser.add_argument("--hype_decimals", type=int, default=18)
    parser.add_argument("--usdt_decimals", type=int, default=6)
    parser.add_argument(
        "--mid_price_column",
        required=False,
        default=None,
        help="Column name in mid-prices CSV to use as mid price (auto-detects if omitted)",
    )

    args = parser.parse_args()
    out_path = main(
        swaps_csv=args.swaps_csv,
        mid_prices_csv=args.mid_prices_csv,
        output_csv=args.output_csv,
        hype_decimals=args.hype_decimals,
        usdt_decimals=args.usdt_decimals,
        mid_price_column=args.mid_price_column,
    )
    print(f"Wrote markout table to: {out_path}")
