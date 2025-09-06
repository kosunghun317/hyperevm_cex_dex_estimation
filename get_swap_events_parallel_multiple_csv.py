#!/usr/bin/env python3
"""
Parallel fetch and decode Swap events for a datetime window, writing one CSV per day.

Overview of parallel distribution and rate limiting:
- The full block window [start_block, end_block] is split into N contiguous,
  non-overlapping segments (one per worker). Example with two workers:
  worker#1 gets [0..1000], worker#2 gets [1001..2000]. This guarantees each
  block is queried exactly once, and segments never overlap.
- Inside each worker, the segment is iterated in sub-chunks of at most
  MAX_BLOCKS_PER_QUERY. If a provider rejects a range, it is adaptively split
  further until acceptable.
- Multiple RPC endpoints are supported via HYPEREVM_RPC_URLS (comma-separated).
  Workers are assigned endpoints in round-robin. Augmentation (tx/block lookups)
  also uses a round-robin pool to spread read load.
- A global rate limiter caps all RPC calls (get_logs, get_block, get_transaction)
  to a fixed requests-per-second budget shared across all threads and endpoints.

Output strategy:
- Instead of accumulating all rows, we stream results as futures complete and
  write to per-day CSV files located in ./event_datasets_final/.
"""

import csv
import json
import os
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from queue import Queue, Empty

from dotenv import load_dotenv
from web3 import Web3
from web3._utils.events import get_event_data
from web3.types import FilterParams, LogReceipt


load_dotenv()


DEFAULT_ADDRESS = "0x337b56d87a6185cd46af3ac2cdf03cbc37070c30"
TOPIC0 = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

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

# Single RPC URL and multi-endpoint list (env vars preferred)
RPC_URL: str = os.environ.get("HYPEREVM_RPC_URL") or "http://localhost:8545"
_URLS_ENV = os.environ.get("HYPEREVM_RPC_URLS", "").strip()
ENDPOINT_URLS: List[str] = (
    [u.strip() for u in _URLS_ENV.split(",") if u.strip()] if _URLS_ENV else [RPC_URL]
)

# Address filter (single, list, or None)
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

# Provider requires small ranges; keep <= 50
MAX_BLOCKS_PER_QUERY: int = 50

# Datetime window (ISO8601)
START_ISO: str = "2025-08-01T00:00:00Z"
END_ISO: str = "2025-08-02T00:00:00Z"

# Output directory for per-day CSVs
OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "event_datasets_final")

# Parallelism
MAX_WORKERS: int = int(os.environ.get("SWAP_EVENTS_MAX_WORKERS", "16"))
Augment_MAX_WORKERS: int = int(
    os.environ.get("SWAP_EVENTS_AUGMENT_WORKERS", str(max(4, MAX_WORKERS)))
)
MAX_AUGMENT_RETRIES: int = int(os.environ.get("SWAP_EVENTS_AUGMENT_RETRIES", "8"))

# Global rate limit (requests per second)
GLOBAL_RPS: int = int(os.environ.get("SWAP_EVENTS_GLOBAL_RPS", "100"))

# Misc
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
    try:
        from web3.middleware import geth_poa_middleware  # type: ignore

        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        return True
    except Exception:
        pass
    try:
        from web3.middleware import ExtraDataToPOAMiddleware  # type: ignore

        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return True
    except Exception:
        pass
    try:
        from web3.middleware.proof_of_authority import (  # type: ignore
            build_proof_of_authority_middleware,
        )

        w3.middleware_onion.inject(build_proof_of_authority_middleware, layer=0)
        return True
    except Exception:
        pass
    return False


def connect_web3(rpc_url: str, use_poa: bool) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if use_poa:
        _inject_poa_middleware_if_available(w3)
    if not w3.is_connected():
        raise RuntimeError("Failed to connect to RPC URL")
    return w3


def _get_block_timestamp(w3: Web3, block_number: int) -> int:
    GLOBAL_LIMITER.acquire()
    return int(w3.eth.get_block(block_number)["timestamp"])  # type: ignore[index]


def _find_block_at_or_after_timestamp(w3: Web3, target_ts: int) -> int:
    latest = int(w3.eth.block_number)
    earliest_ts = _get_block_timestamp(w3, 0)
    if target_ts <= earliest_ts:
        return 0
    latest_ts = _get_block_timestamp(w3, latest)
    if target_ts > latest_ts:
        return latest
    lo, hi, ans = 0, latest, latest
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
    earliest_ts = _get_block_timestamp(w3, 0)
    if target_ts < earliest_ts:
        return 0
    latest_ts = _get_block_timestamp(w3, latest)
    if target_ts >= latest_ts:
        return latest
    lo, hi, ans = 0, latest, 0
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
) -> Iterable[Tuple[int, int]]:
    current = start_block
    while current <= end_block:
        upper = min(current + max_span - 1, end_block)
        yield current, upper
        current = upper + 1


def split_into_segments(
    start_block: int, end_block: int, num_segments: int
) -> List[Tuple[int, int]]:
    if num_segments <= 1:
        return [(start_block, end_block)]
    total = end_block - start_block + 1
    base = total // num_segments
    rem = total % num_segments
    segments: List[Tuple[int, int]] = []
    cur = start_block
    for i in range(num_segments):
        span = base + (1 if i < rem else 0)
        if span <= 0:
            segments.append((cur, cur - 1))
            continue
        seg_start = cur
        seg_end = seg_start + span - 1
        segments.append((seg_start, seg_end))
        cur = seg_end + 1
    return segments


_thread_local = threading.local()


class GlobalRateLimiter:
    def __init__(self, max_per_second: int):
        self.max_per_second = max_per_second
        self._timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self, permits: int = 1) -> None:
        if permits <= 0:
            return
        while True:
            now = monotonic()
            with self._lock:
                while self._timestamps and (now - self._timestamps[0]) >= 1.0:
                    self._timestamps.popleft()
                if len(self._timestamps) + permits <= self.max_per_second:
                    for _ in range(permits):
                        self._timestamps.append(now)
                    return
                oldest = self._timestamps[0] if self._timestamps else now
                wait = max(0.0, 1.0 - (now - oldest))
            sleep(min(0.01, wait if wait > 0 else 0.005))


GLOBAL_LIMITER = GlobalRateLimiter(GLOBAL_RPS)


def _get_thread_w3(endpoint_url: Optional[str] = None) -> Web3:
    w3 = getattr(_thread_local, "w3", None)
    current_url = getattr(_thread_local, "endpoint_url", None)
    target_url = (
        endpoint_url or current_url or (ENDPOINT_URLS[0] if ENDPOINT_URLS else RPC_URL)
    )
    if w3 is None or current_url != target_url:
        w3 = connect_web3(target_url, USE_POA)
        _thread_local.w3 = w3
        _thread_local.endpoint_url = target_url
    return w3


_w3_pool_lock = threading.Lock()
_w3_pool: Optional[List[Web3]] = None
_rr_idx = 0


def _get_round_robin_w3() -> Web3:
    global _w3_pool, _rr_idx
    if _w3_pool is None:
        with _w3_pool_lock:
            if _w3_pool is None:
                _w3_pool = [connect_web3(url, USE_POA) for url in ENDPOINT_URLS]
    with _w3_pool_lock:
        idx = _rr_idx
        _rr_idx = (idx + 1) % len(_w3_pool)  # type: ignore[arg-type]
    return _w3_pool[idx]  # type: ignore[index]


def _adaptive_get_logs(
    w3: Web3,
    base_params: FilterParams,
    from_block: int,
    to_block: int,
) -> List[LogReceipt]:
    params: FilterParams = dict(base_params)
    params["fromBlock"] = from_block
    params["toBlock"] = to_block
    try:
        GLOBAL_LIMITER.acquire()
        return list(w3.eth.get_logs(params))
    except Exception as exc:
        message = str(exc)
        if (
            "query returned more than" in message
            or "Log response size exceeded" in message
            or "response size exceeded" in message
            or "exceeds the limit" in message
        ) and from_block < to_block:
            mid = (from_block + to_block) // 2
            left = _adaptive_get_logs(w3, base_params, from_block, mid)
            right = _adaptive_get_logs(w3, base_params, mid + 1, to_block)
            return left + right
        raise


def _decode_swap_log(w3: Web3, log: LogReceipt) -> Dict[str, Any]:
    decoded = get_event_data(w3.codec, SWAP_EVENT_ABI, log)
    args = decoded["args"]
    topics_list = list(log.get("topics", []))  # type: ignore[assignment]
    row: Dict[str, Any] = {
        "blockNumber": int(decoded.get("blockNumber")),
        "transactionHash": decoded.get("transactionHash").hex()
        if decoded.get("transactionHash") is not None
        else None,
        "logIndex": int(decoded.get("logIndex"))
        if decoded.get("logIndex") is not None
        else None,
        "address": decoded.get("address"),
        "topic0": topics_list[0].hex()
        if len(topics_list) > 0 and hasattr(topics_list[0], "hex")
        else (topics_list[0] if len(topics_list) > 0 else None),
    }
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
            try:
                row[key] = int(args[key])
            except Exception:
                row[key] = args[key]
    return row


def _build_filter_params(
    address_filter: Optional[Union[str, List[str]]],
) -> FilterParams:
    params: FilterParams = {"topics": [TOPIC0]}
    if address_filter:
        if isinstance(address_filter, list):
            params["address"] = [Web3.to_checksum_address(a) for a in address_filter]
        else:
            params["address"] = Web3.to_checksum_address(address_filter)
    return params


def _worker_fetch_range(
    range_pair: Tuple[int, int],
    address_filter: Optional[Union[str, List[str]]],
    endpoint_url: Optional[str],
) -> List[Dict[str, Any]]:
    w3 = _get_thread_w3(endpoint_url)
    start_block, end_block = range_pair
    base_params = _build_filter_params(address_filter)
    rows: List[Dict[str, Any]] = []
    for sub_start, sub_end in chunked_ranges(
        start_block, end_block, MAX_BLOCKS_PER_QUERY
    ):
        logs = _adaptive_get_logs(w3, base_params, sub_start, sub_end)
        if DEBUG:
            if not address_filter:
                addr_str = "<any>"
            elif isinstance(address_filter, list):
                addr_str = f"<list:{len(address_filter)}>"
            else:
                addr_str = address_filter
            print(
                f"Fetched {len(logs)} logs in range [{sub_start}, {sub_end}] for address={addr_str}"
            )
        for log in logs:
            try:
                rows.append(_decode_swap_log(w3, log))
            except Exception as exc:
                print(
                    f"Failed to decode log at block {log['blockNumber']} index {log.get('logIndex')}: {exc}",
                    file=sys.stderr,
                )
    return rows


def _augment_with_tx_and_timestamp(
    w3: Web3,
    row: Dict[str, Any],
    tx_cache: Dict[str, Dict[str, Any]],
    block_ts_cache: Dict[int, int],
) -> Dict[str, Any]:
    tx_hash_hex = row.get("transactionHash")
    if tx_hash_hex:
        if tx_hash_hex in tx_cache:
            tx_info = tx_cache[tx_hash_hex]
        else:
            try:
                GLOBAL_LIMITER.acquire()
                tx = w3.eth.get_transaction(tx_hash_hex)
            except Exception:
                tx = None
            tx_info = {
                "from": Web3.to_checksum_address(tx["from"])
                if tx and tx.get("from")
                else None,
                "to": Web3.to_checksum_address(tx["to"])
                if tx and tx.get("to")
                else None,
            }
            tx_cache[tx_hash_hex] = tx_info
        tx_to = tx_info.get("to") or ZERO_ADDRESS
    else:
        tx_to = ZERO_ADDRESS

    bnum = int(row.get("blockNumber")) if row.get("blockNumber") is not None else None
    ts_val: Optional[int] = None
    if bnum is not None:
        if bnum in block_ts_cache:
            ts_val = block_ts_cache[bnum]
        else:
            try:
                GLOBAL_LIMITER.acquire()
                ts_val = int(w3.eth.get_block(bnum)["timestamp"])  # type: ignore[index]
                block_ts_cache[bnum] = ts_val
            except Exception:
                ts_val = None

    out: Dict[str, Any] = {
        "timestamp": ts_val,
        "block_number": bnum,
        "tx_to": tx_to,
        "amount0": row.get("amount0"),
        "amount1": row.get("amount1"),
        "pool_address": row.get("address"),
    }
    return out


def _stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _ensure_writer(
    day_str: str,
    addr_suffix: str,
    writers: Dict[str, Any],
    files: Dict[str, Any],
    selected_fieldnames: List[str],
) -> Any:
    if day_str in writers:
        return writers[day_str]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"swap_events_parallel_{addr_suffix}_{day_str}.csv")
    f = open(path, "w", encoding="utf-8", newline="")
    files[day_str] = f
    writer = csv.writer(f)
    writer.writerow(selected_fieldnames)
    writers[day_str] = writer
    if DEBUG:
        print(f"Created writer for {day_str} -> {path}")
    return writer


def _writer_worker(
    write_queue: Queue,
    addr_suffix: str,
    selected_fieldnames: List[str],
    csv_batch_size: int,
) -> None:
    writers: Dict[str, Any] = {}
    files: Dict[str, Any] = {}
    day_buffers: Dict[str, List[List[str]]] = {}
    try:
        while True:
            try:
                item = write_queue.get(timeout=1)
            except Empty:
                continue
            if item is None:
                break
            day_str, out_row = item
            buf = day_buffers.get(day_str)
            if buf is None:
                buf = []
                day_buffers[day_str] = buf
            buf.append(out_row)
            if len(buf) >= csv_batch_size:
                writer = _ensure_writer(
                    day_str, addr_suffix, writers, files, selected_fieldnames
                )
                writer.writerows(buf)
                buf.clear()
    finally:
        for day_str, buf in day_buffers.items():
            if not buf:
                continue
            writer = _ensure_writer(
                day_str, addr_suffix, writers, files, selected_fieldnames
            )
            writer.writerows(buf)
            buf.clear()
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass


def main() -> None:
    if not RPC_URL and not ENDPOINT_URLS:
        raise RuntimeError(
            "RPC_URL is not set. Edit the CONFIG section in this script."
        )

    # Use the first endpoint for mapping timestamp->block bounds
    w3 = connect_web3(ENDPOINT_URLS[0], USE_POA)

    start_ts = _parse_iso8601_to_epoch_seconds(START_ISO)
    end_ts = _parse_iso8601_to_epoch_seconds(END_ISO)
    if end_ts < start_ts:
        raise ValueError("END_ISO must be >= START_ISO")

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
            f"Window: {START_ISO} .. {END_ISO} \t blocks [{start_block}, {end_block}] (span={end_block - start_block + 1})"
        )

    # Address filter resolution
    if ADDRESS:
        address_filter: Optional[Union[str, List[str]]] = ADDRESS
    elif ADDRESSES:
        address_filter = ADDRESSES
    else:
        address_filter = None

    # Build addr suffix
    if ADDRESS:
        addr_suffix = ADDRESS[-8:]
    elif ADDRESSES and len(ADDRESSES) > 0:
        addr_suffix = f"list{len(ADDRESSES)}_{ADDRESSES[0][-4:]}"
    else:
        addr_suffix = f"any_{TOPIC0[2:10]}"

    selected_fieldnames: List[str] = [
        "timestamp",
        "block_number",
        "tx_to",
        "amount0",
        "amount1",
        "pool_address",
        "gas_price",
        "gas_used",
    ]

    # Stream-friendly: schedule work per small subrange (<= MAX_BLOCKS_PER_QUERY)
    # to start augmentation/writing early without waiting for whole segments.
    ranges: List[Tuple[int, int]] = list(
        chunked_ranges(start_block, end_block, MAX_BLOCKS_PER_QUERY)
    )

    # Caches and writer queue
    tx_cache: Dict[str, Dict[str, Any]] = {}
    block_ts_cache: Dict[int, int] = {}
    receipt_cache: Dict[str, Dict[str, Optional[int]]] = {}
    total = 0
    CSV_BATCH_SIZE: int = int(os.environ.get("SWAP_EVENTS_CSV_BATCH", "1000"))
    # Unbounded queue so producers never block on I/O
    write_queue: Queue = Queue()
    writer_thread = threading.Thread(
        target=_writer_worker,
        args=(write_queue, addr_suffix, selected_fieldnames, CSV_BATCH_SIZE),
        daemon=True,
    )
    writer_thread.start()
    cache_lock = threading.Lock()

    def augment_task(
        base_row: Dict[str, Any],
    ) -> Optional[Tuple[str, List[str], Optional[int], Optional[str]]]:
        # Double-checked locking strategy for caches inside augmentation
        nonlocal tx_cache, block_ts_cache, receipt_cache
        try:
            # Fast path uses caches if present (no lock on read)
            tx_hash_hex = base_row.get("transactionHash")
            bnum_val = (
                int(base_row.get("blockNumber"))
                if base_row.get("blockNumber") is not None
                else None
            )
            tx_to_val: Optional[str] = None
            ts_val: Optional[int] = None
            gas_price_val: Optional[int] = None
            gas_used_val: Optional[int] = None

            if tx_hash_hex and tx_hash_hex in tx_cache:
                tx_to_val = tx_cache[tx_hash_hex].get("to") or ZERO_ADDRESS
            if bnum_val is not None and bnum_val in block_ts_cache:
                ts_val = block_ts_cache[bnum_val]
            if tx_hash_hex and tx_hash_hex in receipt_cache:
                gas_used_val = receipt_cache[tx_hash_hex].get("gasUsed")
                gas_price_val = receipt_cache[tx_hash_hex].get("gasPrice")

            # Misses → fetch, then write to caches under lock
            if tx_hash_hex and (
                tx_to_val is None or gas_price_val is None or gas_used_val is None
            ):
                last_exc: Optional[Exception] = None
                for _ in range(MAX_AUGMENT_RETRIES):
                    w3_aug = _get_round_robin_w3()
                    try:
                        GLOBAL_LIMITER.acquire()
                        tx = w3_aug.eth.get_transaction(tx_hash_hex)
                        tx_to_val = (
                            Web3.to_checksum_address(tx["to"])
                            if tx and tx.get("to")
                            else ZERO_ADDRESS
                        )
                        tx_gas_price = tx.get("gasPrice") if tx else None
                        GLOBAL_LIMITER.acquire()
                        receipt = w3_aug.eth.get_transaction_receipt(tx_hash_hex)
                        gas_used_val = (
                            int(receipt.get("gasUsed"))
                            if receipt and receipt.get("gasUsed") is not None
                            else None
                        )
                        eff = receipt.get("effectiveGasPrice") if receipt else None
                        gas_price_val = (
                            int(tx_gas_price)
                            if tx_gas_price is not None
                            else (int(eff) if eff is not None else None)
                        )
                        with cache_lock:
                            tx_cache[tx_hash_hex] = {"from": None, "to": tx_to_val}
                            receipt_cache[tx_hash_hex] = {
                                "gasUsed": gas_used_val,
                                "gasPrice": gas_price_val,
                            }
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        continue
                if last_exc is not None and DEBUG:
                    print(
                        f"Augment tx/receipt failed after retries for {tx_hash_hex}: {last_exc}",
                        file=sys.stderr,
                    )

            if bnum_val is not None and ts_val is None:
                last_exc2: Optional[Exception] = None
                for _ in range(MAX_AUGMENT_RETRIES):
                    w3_aug2 = _get_round_robin_w3()
                    try:
                        GLOBAL_LIMITER.acquire()
                        ts_val = int(w3_aug2.eth.get_block(bnum_val)["timestamp"])  # type: ignore[index]
                        with cache_lock:
                            block_ts_cache[bnum_val] = ts_val
                        last_exc2 = None
                        break
                    except Exception as exc:
                        last_exc2 = exc
                        continue
                if last_exc2 is not None and DEBUG:
                    print(
                        f"Augment block ts failed after retries for block {bnum_val}: {last_exc2}",
                        file=sys.stderr,
                    )

            # Build output row list in the same order as selected_fieldnames
            out_row = [
                _stringify_csv_value(ts_val),
                _stringify_csv_value(bnum_val),
                _stringify_csv_value(tx_to_val or ZERO_ADDRESS),
                _stringify_csv_value(base_row.get("amount0")),
                _stringify_csv_value(base_row.get("amount1")),
                _stringify_csv_value(base_row.get("address")),
                _stringify_csv_value(gas_price_val),
                _stringify_csv_value(gas_used_val),
            ]

            if ts_val is None:
                day_str = "unknown"
            else:
                try:
                    day_str = datetime.utcfromtimestamp(int(ts_val)).strftime("%Y%m%d")
                except Exception:
                    day_str = "unknown"

            return day_str, out_row, bnum_val, tx_hash_hex
        except Exception as exc:
            print(
                f"Failed to augment row for tx {base_row.get('transactionHash')}: {exc}",
                file=sys.stderr,
            )
            return None

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i, subr in enumerate(ranges):
                endpoint_url = ENDPOINT_URLS[i % len(ENDPOINT_URLS)]
                futures.append(
                    executor.submit(
                        _worker_fetch_range, subr, address_filter, endpoint_url
                    )
                )

            # Stage 2: parallel augmentation pool
            aug_futures: List[Any] = []
            with ThreadPoolExecutor(max_workers=Augment_MAX_WORKERS) as aug_exec:
                for fut in as_completed(futures):
                    try:
                        rows = fut.result()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        print(f"Worker failed: {exc}", file=sys.stderr)
                        rows = []
                    for base_row in rows:
                        aug_futures.append(aug_exec.submit(augment_task, base_row))

                # Consume augmentation results as they complete and batch-write by day
                def _recover_unknown_day(
                    bnum_val_inner: Optional[int],
                ) -> Optional[str]:
                    if bnum_val_inner is None:
                        return None
                    last_exc3: Optional[Exception] = None
                    for _ in range(max(3, MAX_AUGMENT_RETRIES)):
                        w3_tmp = _get_round_robin_w3()
                        try:
                            GLOBAL_LIMITER.acquire()
                            ts_tmp = int(
                                w3_tmp.eth.get_block(bnum_val_inner)["timestamp"]
                            )  # type: ignore[index]
                            with cache_lock:
                                block_ts_cache[bnum_val_inner] = ts_tmp
                            return datetime.utcfromtimestamp(int(ts_tmp)).strftime(
                                "%Y%m%d"
                            )
                        except Exception as exc:
                            last_exc3 = exc
                            continue
                    if DEBUG and last_exc3 is not None:
                        print(
                            f"Failed to recover day for block {bnum_val_inner}: {last_exc3}",
                            file=sys.stderr,
                        )
                    return None

                for afut in as_completed(aug_futures):
                    res = afut.result()
                    if res is None:
                        continue
                    day_str, out_row, bnum_val_inner, _txh = res
                    if day_str == "unknown":
                        recovered = _recover_unknown_day(bnum_val_inner)
                        if recovered is not None:
                            day_str = recovered
                            # Update timestamp string if we cached it
                            if (
                                bnum_val_inner is not None
                                and bnum_val_inner in block_ts_cache
                            ):
                                out_row[0] = _stringify_csv_value(
                                    block_ts_cache[bnum_val_inner]
                                )
                    write_queue.put((day_str, out_row))
                    total += 1

        # Signal writer thread to flush and exit
        write_queue.put(None)
        writer_thread.join()

        print(f"Wrote total {total} rows across day files in {OUTPUT_DIR}")
    finally:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
