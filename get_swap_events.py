#!/usr/bin/env python3
"""
Fetch and decode Swap events from a contract for a datetime window, querying at most
50 blocks per request (or fewer if the provider limits results).

How to use:
- Manually edit the CONFIG section below: `RPC_URL`, `ADDRESS`, `USE_POA`,
  `MAX_BLOCKS_PER_QUERY`, `START_ISO`, `END_ISO`, and optionally `OUTPUT_CSV`.
- Run: `python get_swap_events.py`

Notes:
- The script filters by the provided topic0 and address.
- It decodes logs using the given Swap event ABI.
- It converts the datetime window to a block range via binary search on block timestamps.
- Each request covers at most 50 blocks; if the provider errors with too many results,
  the script automatically reduces the chunk size until it succeeds.
"""

import json
import csv
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Union
from datetime import datetime, timezone

from web3 import Web3
from web3.types import FilterParams, LogReceipt
from web3._utils.events import get_event_data
from dotenv import load_dotenv

load_dotenv()


DEFAULT_ADDRESS = "0x337b56d87a6185cd46af3ac2cdf03cbc37070c30"
TOPIC0 = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Swap event ABI provided by the user
SWAP_EVENT_ABI: Dict[str, Any] = {
    "anonymous": False,
    "inputs": [
        {
            "indexed": True,
            "internalType": "address",
            "name": "sender",
            "type": "address",
        },
        {
            "indexed": True,
            "internalType": "address",
            "name": "recipient",
            "type": "address",
        },
        {
            "indexed": False,
            "internalType": "int256",
            "name": "amount0",
            "type": "int256",
        },
        {
            "indexed": False,
            "internalType": "int256",
            "name": "amount1",
            "type": "int256",
        },
        {
            "indexed": False,
            "internalType": "uint160",
            "name": "sqrtPriceX96",
            "type": "uint160",
        },
        {
            "indexed": False,
            "internalType": "uint128",
            "name": "liquidity",
            "type": "uint128",
        },
        {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"},
    ],
    "name": "Swap",
    "type": "event",
}


# =====================
# CONFIG (edit these)
# =====================

# RPC endpoint URL (set to env var HYPEREVM_RPC_URL if present, otherwise hardcode here)
RPC_URL: str = os.environ.get("HYPEREVM_RPC_URL") or "http://localhost:8545"

# Contract address to filter logs (set to None to search all emitters for this topic)
# You can also set ADDRESSES to a list of contracts to query them together.
ADDRESS: Optional[str] = None
ADDRESSES: Optional[List[str]] = [
    "0xBd19E19E4b70eB7F248695a42208bc1EdBBFb57D",
    "0x161fB7d6c764f81DAE581E8a4981772750416727",
    "0x337b56d87a6185cd46af3ac2cdf03cbc37070c30",
    "0x56abfaf40f5b7464e9cc8cff1af13863d6914508",
    "0xf40d57783c3359f160d006b9bc7a2e4311fe6a86",
    "0x68f4Bb01D75Edf476195fCA80271ADf5CA71A242",
    "0x3603ffEbB994CC110b4186040CaC3005B2cf4465",
]
USE_POA: bool = False

# Maximum number of blocks per query (must be <= 50 per user requirement)
MAX_BLOCKS_PER_QUERY: int = 50

# Datetime window (ISO8601). Example: "2025-08-01T00:00:00Z"
START_ISO: str = "2025-08-01T00:00:00Z"
END_ISO: str = "2025-08-31T00:00:00Z"

# Output directory and file (CSV). If OUTPUT_CSV is None, a filename will be generated.
OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "event_datasets")
OUTPUT_CSV: Optional[str] = None

# Debug prints
DEBUG: bool = True


def _parse_iso8601_to_epoch_seconds(iso_str: str) -> int:
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _inject_poa_middleware_if_available(w3: Web3) -> bool:
    """Try to inject a POA-compatible middleware for different web3.py versions.

    Returns True if a middleware was injected, False otherwise.
    """
    # web3.py v5 style
    try:
        from web3.middleware import geth_poa_middleware  # type: ignore

        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        return True
    except Exception:
        pass

    # web3.py v6 style (ExtraDataToPOAMiddleware)
    try:
        from web3.middleware import ExtraDataToPOAMiddleware  # type: ignore

        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return True
    except Exception:
        pass

    # Alternate POA builder
    try:
        from web3.middleware.proof_of_authority import (
            build_proof_of_authority_middleware,
        )  # type: ignore

        w3.middleware_onion.inject(build_proof_of_authority_middleware, layer=0)
        return True
    except Exception:
        pass

    return False


def connect_web3(rpc_url: str, use_poa: bool) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if use_poa:
        injected = _inject_poa_middleware_if_available(w3)
        if not injected:
            print(
                "POA middleware not available in this web3 version; continuing without it",
                file=sys.stderr,
            )
    if not w3.is_connected():
        raise RuntimeError("Failed to connect to RPC URL")
    return w3


def _augment_with_tx_context(
    w3: Web3,
    row: Dict[str, Any],
    tx_cache: Dict[str, Dict[str, Any]],
) -> None:
    """Populate only tx_from and tx_to using a single transaction lookup.

    If tx.to is missing (contract creation), leave tx_to as the zero address
    without attempting any receipt-based resolution.
    """
    tx_hash_hex = row.get("transactionHash")
    if not tx_hash_hex:
        return

    # Fetch transaction (cached)
    if tx_hash_hex in tx_cache:
        tx_info = tx_cache[tx_hash_hex]
    else:
        try:
            tx = w3.eth.get_transaction(tx_hash_hex)
        except Exception:
            tx = None
        tx_info = {
            "from": Web3.to_checksum_address(tx["from"])
            if tx and tx.get("from")
            else None,
            "to": Web3.to_checksum_address(tx["to"]) if tx and tx.get("to") else None,
        }
        tx_cache[tx_hash_hex] = tx_info

    # Set fields; default tx_to to zero address if missing
    row["tx_from"] = tx_info.get("from")
    row["tx_to"] = tx_info.get("to") or ZERO_ADDRESS


def _get_block_timestamp(w3: Web3, block_number: int) -> int:
    return int(w3.eth.get_block(block_number)["timestamp"])  # type: ignore[index]


def _find_block_at_or_after_timestamp(w3: Web3, target_ts: int) -> int:
    latest = int(w3.eth.block_number)
    # Fast-path bounds
    earliest_ts = _get_block_timestamp(w3, 0)
    if target_ts <= earliest_ts:
        return 0
    latest_ts = _get_block_timestamp(w3, latest)
    if target_ts > latest_ts:
        return latest

    lo = 0
    hi = latest
    ans = latest
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = _get_block_timestamp(w3, mid)
        if ts >= target_ts:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def _find_block_at_or_before_timestamp(w3: Web3, target_ts: int) -> int:
    latest = int(w3.eth.block_number)
    # Fast-path bounds
    earliest_ts = _get_block_timestamp(w3, 0)
    if target_ts < earliest_ts:
        return 0
    latest_ts = _get_block_timestamp(w3, latest)
    if target_ts >= latest_ts:
        return latest

    lo = 0
    hi = latest
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = _get_block_timestamp(w3, mid)
        if ts <= target_ts:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def chunked_ranges(
    start_block: int, end_block: int, max_span: int
) -> Iterable[tuple[int, int]]:
    current = start_block
    while current <= end_block:
        end = min(current + max_span - 1, end_block)
        yield current, end
        current = end + 1


def fetch_logs_chunked(
    w3: Web3,
    address: Optional[Union[str, List[str]]],
    topic0: str,
    start_block: int,
    end_block: int,
    max_blocks_per_query: int,
) -> Iterable[LogReceipt]:
    # Adaptive chunking: start with max_blocks_per_query, reduce on provider errors
    current = start_block
    step = max_blocks_per_query
    while current <= end_block:
        upper = min(current + step - 1, end_block)
        params: FilterParams = {
            "fromBlock": current,
            "toBlock": upper,
            "topics": [topic0],
        }
        if address:
            if isinstance(address, list):
                params["address"] = [Web3.to_checksum_address(a) for a in address]
            else:
                params["address"] = Web3.to_checksum_address(address)
        try:
            logs = w3.eth.get_logs(params)
            if DEBUG:
                if not address:
                    addr_str = "<any>"
                elif isinstance(address, list):
                    addr_str = f"<list:{len(address)}>"
                else:
                    addr_str = address
                print(
                    f"Fetched {len(logs)} logs in range [{current}, {upper}] for address={addr_str}"
                )
        except Exception as exc:
            message = str(exc)
            # Reduce chunk size on typical provider volume errors
            if (
                "query returned more than" in message
                or "Log response size exceeded" in message
                or "response size exceeded" in message
                or "exceeds the limit" in message
            ) and step > 1:
                step = max(1, step // 2)
                continue
            raise

        for log in logs:
            yield log

        # Move to next range; try to gently scale up if we had previously shrunk
        current = upper + 1
        if step < max_blocks_per_query:
            step = min(max_blocks_per_query, step * 2)


def decode_swap_log(w3: Web3, log: LogReceipt) -> Dict[str, Any]:
    """Decode a Swap log and return a flattened dictionary with all useful fields.

    Includes:
    - Core log metadata: blockNumber, blockHash, transactionHash, transactionIndex, logIndex, address, removed
    - Event info: event name
    - Decoded args flattened as top-level keys (sender, recipient, amount0, ...)
    - Raw payload: data, topic0..topic3, topics_json
    - Transaction participants (added later): tx_from (EOA), tx_to
    """
    decoded = get_event_data(w3.codec, SWAP_EVENT_ABI, log)
    args = decoded["args"]

    topics_list = list(log.get("topics", []))  # type: ignore[assignment]
    row: Dict[str, Any] = {
        "blockNumber": int(decoded.get("blockNumber")),
        "blockHash": decoded.get("blockHash").hex()
        if decoded.get("blockHash") is not None
        else None,
        "transactionHash": decoded.get("transactionHash").hex()
        if decoded.get("transactionHash") is not None
        else None,
        "transactionIndex": int(decoded.get("transactionIndex"))
        if decoded.get("transactionIndex") is not None
        else None,
        "logIndex": int(decoded.get("logIndex"))
        if decoded.get("logIndex") is not None
        else None,
        "address": decoded.get("address"),
        "event": decoded.get("event"),
        "removed": bool(log.get("removed", False)),
        "data": log.get("data"),
        "topics_json": json.dumps(
            [t.hex() if isinstance(t, (bytes, bytearray)) else t for t in topics_list],
            separators=(",", ":"),
        ),
        "topic0": topics_list[0].hex()
        if len(topics_list) > 0 and hasattr(topics_list[0], "hex")
        else (topics_list[0] if len(topics_list) > 0 else None),
        "topic1": topics_list[1].hex()
        if len(topics_list) > 1 and hasattr(topics_list[1], "hex")
        else (topics_list[1] if len(topics_list) > 1 else None),
        "topic2": topics_list[2].hex()
        if len(topics_list) > 2 and hasattr(topics_list[2], "hex")
        else (topics_list[2] if len(topics_list) > 2 else None),
        "topic3": topics_list[3].hex()
        if len(topics_list) > 3 and hasattr(topics_list[3], "hex")
        else (topics_list[3] if len(topics_list) > 3 else None),
    }

    # Flatten args
    for key in [
        "sender",
        "recipient",
        "amount0",
        "amount1",
        "sqrtPriceX96",
        "liquidity",
        "tick",
    ]:
        if key in args:
            val = args[key]
            try:
                row[key] = int(val)
            except Exception:
                row[key] = val

    # Add raw payloads for completeness
    def _normalize_for_json(obj: Any) -> Any:
        if isinstance(obj, (bytes, bytearray)):
            return obj.hex()
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: _normalize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_normalize_for_json(v) for v in obj]
        try:
            # web3 AttributeDict and HexBytes handling
            d = dict(obj)  # type: ignore[arg-type]
            return {k: _normalize_for_json(v) for k, v in d.items()}
        except Exception:
            return str(obj)

    row["raw_log_json"] = json.dumps(_normalize_for_json(log), separators=(",", ":"))
    row["decoded_args_json"] = json.dumps(
        _normalize_for_json(dict(args)), separators=(",", ":")
    )

    return row


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def main() -> None:
    # Validate configuration
    if not RPC_URL:
        raise RuntimeError(
            "RPC_URL is not set. Edit the CONFIG section in this script."
        )

    # Connect
    w3 = connect_web3(RPC_URL, USE_POA)

    # Convert datetimes to epoch seconds
    start_ts = _parse_iso8601_to_epoch_seconds(START_ISO)
    end_ts = _parse_iso8601_to_epoch_seconds(END_ISO)
    if end_ts < start_ts:
        raise ValueError("END_ISO must be >= START_ISO")

    # Map timestamps to block numbers
    start_block = _find_block_at_or_after_timestamp(w3, start_ts)
    end_block = _find_block_at_or_before_timestamp(w3, end_ts)
    if end_block < start_block:
        print("No blocks in the specified datetime window", file=sys.stderr)
        return

    if DEBUG:
        try:
            client_version = w3.client_version
        except Exception:
            client_version = "unknown"
        try:
            chain_id = w3.eth.chain_id
        except Exception:
            chain_id = "unknown"
        print(
            f"Connected to chain_id={chain_id}, client='{client_version}'. Latest block={w3.eth.block_number}."
        )
        print(
            f"Window: {START_ISO} .. {END_ISO} → blocks [{start_block}, {end_block}] (span={end_block - start_block + 1})"
        )
        if ADDRESS or ADDRESSES:
            if ADDRESS:
                checksum_addr = Web3.to_checksum_address(ADDRESS)
                code = w3.eth.get_code(checksum_addr)
                print(
                    f"Address: {checksum_addr} (code_size={len(code) if code is not None else 0})"
                )
            if ADDRESSES:
                addrs = [Web3.to_checksum_address(a) for a in ADDRESSES]
                print(f"Addresses: {len(addrs)} configured")
        else:
            print("Address: <any>")
        # Compare topic from ABI vs configured TOPIC0
        try:
            from eth_utils import keccak

            signature = (
                "Swap(address,address,int256,int256,uint160,uint128,int24)".encode()
            )
            computed_topic0 = (
                "0x"
                + keccak(
                    text="Swap(address,address,int256,int256,uint160,uint128,int24)"
                ).hex()
            )
            print(
                f"Configured topic0={TOPIC0}; computed from ABI signature={computed_topic0}"
            )
        except Exception:
            pass

        # Quick probe without address filter on first chunk
        probe_to = min(start_block + MAX_BLOCKS_PER_QUERY - 1, end_block)
        try:
            probe_logs = w3.eth.get_logs(
                {
                    "fromBlock": start_block,
                    "toBlock": probe_to,
                    "topics": [TOPIC0],
                }
            )
            print(
                f"Probe any-address topic-only: {len(probe_logs)} logs in [{start_block}, {probe_to}]"
            )
            # Summarize unique emitters
            unique_addrs = {}
            for l in probe_logs[:50]:  # cap for speed
                unique_addrs[l["address"]] = unique_addrs.get(l["address"], 0) + 1
            if unique_addrs:
                top = sorted(unique_addrs.items(), key=lambda kv: kv[1], reverse=True)[
                    :10
                ]
                print("Emitters in probe (top 10):")
                for a, c in top:
                    print(f"  {a}: {c} logs")
        except Exception as exc:
            print(f"Probe topic-only failed: {exc}", file=sys.stderr)

    # Resolve output path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if OUTPUT_CSV is None:
        # Build a simple, safe filename based on dates and address suffix
        def _safe_date(s: str) -> str:
            return (
                s.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
            )

        if ADDRESS:
            addr_suffix = ADDRESS[-8:]
        elif ADDRESSES and len(ADDRESSES) > 0:
            addr_suffix = f"list{len(ADDRESSES)}_{ADDRESSES[0][-4:]}"
        else:
            addr_suffix = f"any_{TOPIC0[2:10]}"
        OUTPUT_PATH = os.path.join(
            OUTPUT_DIR,
            f"swap_events_{addr_suffix}_{_safe_date(START_ISO)}_{_safe_date(END_ISO)}.csv",
        )
    else:
        OUTPUT_PATH = OUTPUT_CSV

    total = 0
    # Only export these selected columns as requested
    selected_fieldnames: List[str] = [
        "timestamp",
        "block_number",
        "tx_to",
        "amount0",
        "amount1",
        "pool_address",
    ]
    # Caches to reduce RPC calls
    block_ts_cache: Dict[int, int] = {}
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=selected_fieldnames, extrasaction="ignore"
            )
            writer.writeheader()

            # Choose address filter (single, list, or None)
            address_filter: Optional[Union[str, List[str]]]
            if ADDRESS:
                address_filter = ADDRESS
            elif ADDRESSES:
                address_filter = ADDRESSES
            else:
                address_filter = None

            for log in fetch_logs_chunked(
                w3=w3,
                address=address_filter,
                topic0=TOPIC0,
                start_block=start_block,
                end_block=end_block,
                max_blocks_per_query=MAX_BLOCKS_PER_QUERY,
            ):
                try:
                    decoded = decode_swap_log(w3, log)
                except (
                    Exception
                ) as exc:  # decoding issues should not stop the whole run
                    print(
                        f"Failed to decode log at block {log['blockNumber']} index {log['logIndex']}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                # Augment with transaction context (EOA sender and first receiver)
                try:
                    # Initialize cache once
                    if "tx_cache" not in locals():
                        tx_cache: Dict[str, Dict[str, Any]] = {}
                    _augment_with_tx_context(w3, decoded, tx_cache)
                except Exception as exc:
                    print(
                        f"Failed to add tx context for {decoded.get('transactionHash')}: {exc}",
                        file=sys.stderr,
                    )

                # Resolve block timestamp with cache
                bnum = (
                    int(decoded.get("blockNumber"))
                    if decoded.get("blockNumber") is not None
                    else None
                )
                ts_val: Optional[int] = None
                if bnum is not None:
                    if bnum in block_ts_cache:
                        ts_val = block_ts_cache[bnum]
                    else:
                        try:
                            ts_val = int(w3.eth.get_block(bnum)["timestamp"])  # type: ignore[index]
                            block_ts_cache[bnum] = ts_val
                        except Exception:
                            ts_val = None

                row_out: Dict[str, Any] = {
                    "timestamp": ts_val,
                    "block_number": bnum,
                    "tx_to": decoded.get("tx_to"),
                    "amount0": decoded.get("amount0"),
                    "amount1": decoded.get("amount1"),
                    "pool_address": decoded.get("address"),
                }

                writer.writerow(
                    {
                        k: _stringify_csv_value(row_out.get(k))
                        for k in selected_fieldnames
                    }
                )
                total += 1

        print(f"Wrote {total} rows to {OUTPUT_PATH}")

    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()
