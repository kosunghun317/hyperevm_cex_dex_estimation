#!/usr/bin/env python3
"""
Async, high-parallelism, non-blocking pipeline to fetch Swap events and write per-day CSVs.

Key differences vs threaded approach:
- Uses asyncio + aiohttp to maximize in-flight JSON-RPC requests.
- Global token-bucket limiter to respect total RPS across all endpoints.
- Per-endpoint concurrency caps to avoid overloading any single provider.
- Non-blocking file I/O via a dedicated writer thread with an unbounded Queue.
- Direct ABI decoding of event `data` to avoid heavy web3 event decoding.

Dependencies: aiohttp (pip install aiohttp)
"""

import asyncio
import csv
import json
import os
import sys
import threading
import shutil
from collections import deque
from datetime import datetime, timezone
from queue import Queue
from time import monotonic
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

try:
    import aiohttp
except Exception as exc:
    print(
        "This script requires aiohttp. Install with: pip install aiohttp",
        file=sys.stderr,
    )
    raise


# =====================
# CONFIG (env-overridable)
# =====================

RPC_URLS_ENV = os.environ.get("HYPEREVM_RPC_URLS", "").strip()
ENDPOINT_URLS: List[str] = [
    u.strip() for u in RPC_URLS_ENV.split(",") if u.strip()
] or [os.environ.get("HYPEREVM_RPC_URL") or "http://localhost:8545"]

# Addresses filter (single or list). If empty, topic-only search across all pools
ADDRESSES: Optional[List[str]] = [
    "0xBd19E19E4b70eB7F248695a42208bc1EdBBFb57D",
    "0x161fB7d6c764f81DAE581E8a4981772750416727",
    "0x337b56d87a6185cd46af3ac2cdf03cbc37070c30",
    "0x56abfaf40f5b7464e9cc8cff1af13863d6914508",
    "0xf40d57783c3359f160d006b9bc7a2e4311fe6a86",
    "0x68f4Bb01D75Edf476195fCA80271ADf5CA71A242",
    "0x3603ffEbB994CC110b4186040CaC3005B2cf4465",
]

TOPIC0 = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"  # Swap

MAX_BLOCKS_PER_QUERY = int(os.environ.get("SWAP_EVENTS_MAX_BLOCKS", "50"))
GLOBAL_RPS = min(100, int(os.environ.get("SWAP_EVENTS_GLOBAL_RPS", "100")))
PER_ENDPOINT_CONCURRENCY = int(os.environ.get("SWAP_EVENTS_ENDPOINT_CONCURRENCY", "8"))
AUGMENT_CONCURRENCY = int(os.environ.get("SWAP_EVENTS_AUGMENT_CONCURRENCY", "16"))
CSV_BATCH_SIZE = int(os.environ.get("SWAP_EVENTS_CSV_BATCH", "1_000"))

START_ISO = os.environ.get("SWAP_EVENTS_START_ISO", "2025-08-16T00:00:00Z")
END_ISO = os.environ.get("SWAP_EVENTS_END_ISO", "2025-08-31T00:00:00Z")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "event_datasets_final_async")

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEBUG = os.environ.get("SWAP_EVENTS_DEBUG", "1") == "1"
DEFER_WRITE = os.environ.get("SWAP_EVENTS_DEFER_WRITE", "0") == "1"
PROGRESS_SEGMENTS = int(os.environ.get("SWAP_EVENTS_PROGRESS_SEGMENTS", "64"))
PROGRESS_INTERVAL_SEC = float(os.environ.get("SWAP_EVENTS_PROGRESS_INTERVAL", "2.0"))


# =====================
# Helpers
# =====================


def _parse_iso8601_to_epoch_seconds(iso_str: str) -> int:
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _safe_date(s: str) -> str:
    return s.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")


def to_hex_block(n: int) -> str:
    return hex(int(n))


def decode_int256(word_hex: str) -> int:
    b = bytes.fromhex(word_hex)
    return int.from_bytes(b, byteorder="big", signed=True)


def decode_uint(word_hex: str) -> int:
    b = bytes.fromhex(word_hex)
    return int.from_bytes(b, byteorder="big", signed=False)


def decode_swap_data(data_hex: str) -> Tuple[Optional[int], Optional[int]]:
    # data is 0x + 32-byte words concatenated. We need amount0 (int256), amount1 (int256)
    if not data_hex or not data_hex.startswith("0x"):
        return None, None
    payload = data_hex[2:]
    if len(payload) < 64 * 2:
        return None, None
    w0 = payload[0:64]
    w1 = payload[64:128]
    try:
        amount0 = decode_int256(w0)
        amount1 = decode_int256(w1)
        return amount0, amount1
    except Exception:
        return None, None


# =====================
# Rate limiter
# =====================


class AsyncTokenBucket:
    def __init__(self, rate_per_sec: int):
        self.rate = rate_per_sec
        self._ts = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, permits: int = 1) -> None:
        if permits <= 0:
            return
        while True:
            now = monotonic()
            async with self._lock:
                while self._ts and (now - self._ts[0]) >= 1.0:
                    self._ts.popleft()
                if len(self._ts) + permits <= self.rate:
                    for _ in range(permits):
                        self._ts.append(now)
                    return
                # wait until earliest token is older than 1s
                oldest = self._ts[0] if self._ts else now
                wait = max(0.0, 1.0 - (now - oldest))
            await asyncio.sleep(min(0.01, wait if wait > 0 else 0.005))


GLOBAL_LIMITER = AsyncTokenBucket(GLOBAL_RPS)


# =====================
# RPC client
# =====================


class RpcClient:
    def __init__(self, url: str, session: aiohttp.ClientSession, concurrency: int):
        self.url = url
        self.session = session
        self.semaphore = asyncio.Semaphore(concurrency)
        self._id = 0

    async def _rpc(self, method: str, params: List[Any]) -> Any:
        async with self.semaphore:
            await GLOBAL_LIMITER.acquire()
            self._id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self._id,
                "method": method,
                "params": params,
            }
            async with self.session.post(
                self.url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {method}")
                data = await resp.json(loads=json.loads)
                if "error" in data:
                    raise RuntimeError(f"RPC error for {method}: {data['error']}")
                return data.get("result")

    async def eth_blockNumber(self) -> int:
        res = await self._rpc("eth_blockNumber", [])
        return int(res, 16)

    async def eth_getBlockByNumber(self, block: int) -> Dict[str, Any]:
        res = await self._rpc("eth_getBlockByNumber", [to_hex_block(block), False])
        return res

    async def eth_getLogs(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        res = await self._rpc("eth_getLogs", [params])
        return res or []

    async def eth_getTransactionByHash(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        res = await self._rpc("eth_getTransactionByHash", [tx_hash])
        return res

    async def eth_getTransactionReceipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        res = await self._rpc("eth_getTransactionReceipt", [tx_hash])
        return res


# =====================
# Async block mapping
# =====================


async def get_block_timestamp(client: RpcClient, block_number: int) -> int:
    blk = await client.eth_getBlockByNumber(block_number)
    return int(blk["timestamp"], 16)


async def find_block_at_or_after_timestamp(client: RpcClient, target_ts: int) -> int:
    latest = await client.eth_blockNumber()
    earliest_ts = await get_block_timestamp(client, 0)
    if target_ts <= earliest_ts:
        return 0
    latest_ts = await get_block_timestamp(client, latest)
    if target_ts > latest_ts:
        return latest
    lo, hi, ans = 0, latest, latest
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = await get_block_timestamp(client, mid)
        if ts >= target_ts:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


async def find_block_at_or_before_timestamp(client: RpcClient, target_ts: int) -> int:
    latest = await client.eth_blockNumber()
    earliest_ts = await get_block_timestamp(client, 0)
    if target_ts < earliest_ts:
        return 0
    latest_ts = await get_block_timestamp(client, latest)
    if target_ts >= latest_ts:
        return latest
    lo, hi, ans = 0, latest, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = await get_block_timestamp(client, mid)
        if ts <= target_ts:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


# =====================
# Producer: getLogs with adaptive splitting
# =====================


def chunked_ranges(
    start_block: int, end_block: int, max_span: int
) -> Iterable[Tuple[int, int]]:
    current = start_block
    while current <= end_block:
        upper = min(current + max_span - 1, end_block)
        yield current, upper
        current = upper + 1


def build_filter(
    addresses: Optional[List[str]], topic0: str, start_block: int, end_block: int
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "fromBlock": to_hex_block(start_block),
        "toBlock": to_hex_block(end_block),
        "topics": [topic0],
    }
    if addresses:
        params["address"] = [a for a in addresses]
    return params


async def fetch_logs_adaptive(
    client: RpcClient, params: Dict[str, Any], start_block: int, end_block: int
) -> List[Dict[str, Any]]:
    # try full range, on error halve
    try:
        params = dict(params)
        params["fromBlock"] = to_hex_block(start_block)
        params["toBlock"] = to_hex_block(end_block)
        return await client.eth_getLogs(params)
    except Exception as exc:
        msg = str(exc)
        if start_block < end_block and (
            "more than" in msg
            or "exceeded" in msg
            or "limit" in msg
            or "response size" in msg
        ):
            mid = (start_block + end_block) // 2
            left, right = await asyncio.gather(
                fetch_logs_adaptive(client, params, start_block, mid),
                fetch_logs_adaptive(client, params, mid + 1, end_block),
            )
            return left + right
        raise


# =====================
# Augmentation and writing
# =====================


def ensure_writer(
    day_str: str,
    addr_suffix: str,
    writers: Dict[str, Any],
    files: Dict[str, Any],
    header: List[str],
) -> Any:
    if day_str in writers:
        return writers[day_str]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"swap_events_async_{addr_suffix}_{day_str}.csv")
    f = open(path, "w", encoding="utf-8", newline="")
    files[day_str] = f
    writer = csv.writer(f)
    writer.writerow(header)
    writers[day_str] = writer
    # writer creation message suppressed to avoid extra output
    return writer


def writer_thread_fn(
    write_q: Queue,
    addr_suffix: str,
    header: List[str],
    batch_size: int,
    defer_write: bool,
    start_block: int,
    seg_size: int,
    write_done: List[bool],
    written_rows_per_seg: List[int],
    enqueued_rows_per_seg: List[int],
    stats: Dict[str, int],
    state_lock: threading.Lock,
    fetch_done: List[bool],
) -> None:
    writers: Dict[str, Any] = {}
    files: Dict[str, Any] = {}
    buffers: Dict[str, List[List[str]]] = {}
    while True:
        item = write_q.get()
        if item is None:
            break
        day_str, row_list = item
        buf = buffers.setdefault(day_str, [])
        buf.append(row_list)
        if not defer_write and len(buf) >= batch_size:
            w = ensure_writer(day_str, addr_suffix, writers, files, header)
            w.writerows(buf)
            # update write stats
            with state_lock:
                for row in buf:
                    try:
                        bnum = int(row[1])
                    except Exception:
                        continue
                    seg_idx = min(
                        PROGRESS_SEGMENTS - 1, max(0, (bnum - start_block) // seg_size)
                    )
                    written_rows_per_seg[seg_idx] += 1
                    if (
                        fetch_done[seg_idx]
                        and written_rows_per_seg[seg_idx]
                        >= enqueued_rows_per_seg[seg_idx]
                    ):
                        write_done[seg_idx] = True
                stats["written_rows"] = stats.get("written_rows", 0) + len(buf)
            buf.clear()
    # flush
    for day_str, buf in buffers.items():
        if not buf:
            continue
        w = ensure_writer(day_str, addr_suffix, writers, files, header)
        w.writerows(buf)
        with state_lock:
            for row in buf:
                try:
                    bnum = int(row[1])
                except Exception:
                    continue
                seg_idx = min(
                    PROGRESS_SEGMENTS - 1, max(0, (bnum - start_block) // seg_size)
                )
                written_rows_per_seg[seg_idx] += 1
                if (
                    fetch_done[seg_idx]
                    and written_rows_per_seg[seg_idx] >= enqueued_rows_per_seg[seg_idx]
                ):
                    write_done[seg_idx] = True
            stats["written_rows"] = stats.get("written_rows", 0) + len(buf)
        buf.clear()
    for f in files.values():
        try:
            f.close()
        except Exception:
            pass


async def augment_and_enqueue(
    logs: List[Dict[str, Any]],
    clients: List[RpcClient],
    out_q: Queue,
    augment_sem: asyncio.Semaphore,
    block_ts_cache: Dict[int, int],
    tx_cache: Dict[str, Tuple[Optional[str], Optional[int], Optional[int]]],
) -> None:
    async def _augment_one(log: Dict[str, Any]) -> None:
        async with augment_sem:
            block_number = (
                int(log.get("blockNumber"), 16)
                if isinstance(log.get("blockNumber"), str)
                else int(log["blockNumber"])
            )  # web3 returns hex, raw RPC returns hex
            tx_hash = log.get("transactionHash")
            pool_address = log.get("address")
            data_hex = log.get("data")

            amount0, amount1 = decode_swap_data(data_hex)

            # timestamp via cache
            ts_val = block_ts_cache.get(block_number)
            if ts_val is None:
                client = clients[block_number % len(clients)]
                blk = await client.eth_getBlockByNumber(block_number)
                ts_val = int(blk["timestamp"], 16)
                block_ts_cache[block_number] = ts_val

            tx_to: Optional[str] = None
            gas_price: Optional[int] = None
            gas_used: Optional[int] = None
            if isinstance(tx_hash, (str, bytes)):
                key = tx_hash if isinstance(tx_hash, str) else tx_hash.hex()
                cached = tx_cache.get(key)
                if cached is None:
                    client = clients[block_number % len(clients)]
                    tx = await client.eth_getTransactionByHash(key)
                    rcpt = await client.eth_getTransactionReceipt(key)
                    if tx is not None:
                        to_field = tx.get("to")
                        tx_to = (
                            to_field
                            if isinstance(to_field, str) and to_field
                            else ZERO_ADDRESS
                        )
                        gp = tx.get("gasPrice")
                        gas_price = (
                            int(gp, 16)
                            if isinstance(gp, str)
                            else (int(gp) if gp is not None else None)
                        )
                    if rcpt is not None:
                        gu = rcpt.get("gasUsed")
                        gas_used = (
                            int(gu, 16)
                            if isinstance(gu, str)
                            else (int(gu) if gu is not None else None)
                        )
                        if gas_price is None:
                            egp = rcpt.get("effectiveGasPrice")
                            gas_price = (
                                int(egp, 16)
                                if isinstance(egp, str)
                                else (int(egp) if egp is not None else None)
                            )
                    tx_cache[key] = (tx_to, gas_price, gas_used)
                else:
                    tx_to, gas_price, gas_used = cached

            day_str = (
                datetime.utcfromtimestamp(ts_val).strftime("%Y%m%d")
                if ts_val is not None
                else "unknown"
            )
            row = [
                str(ts_val if ts_val is not None else ""),
                str(block_number),
                tx_to or ZERO_ADDRESS,
                str(amount0 if amount0 is not None else ""),
                str(amount1 if amount1 is not None else ""),
                pool_address,
                str(gas_price if gas_price is not None else ""),
                str(gas_used if gas_used is not None else ""),
            ]
            out_q.put((day_str, row))

    await asyncio.gather(*[_augment_one(l) for l in logs])


# =====================
# Main
# =====================


async def main_async() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    header = [
        "timestamp",
        "block_number",
        "tx_to",
        "amount0",
        "amount1",
        "pool_address",
        "gas_price",
        "gas_used",
    ]

    # Address suffix for filenames
    if ADDRESSES and len(ADDRESSES) > 0:
        addr_suffix = f"list{len(ADDRESSES)}_{ADDRESSES[0][-4:]}"
    else:
        addr_suffix = f"any_{TOPIC0[2:10]}"

    write_q: Queue = Queue()
    # Writer thread is started after progress setup (below)

    start_ts = _parse_iso8601_to_epoch_seconds(START_ISO)
    end_ts = _parse_iso8601_to_epoch_seconds(END_ISO)
    if end_ts < start_ts:
        raise RuntimeError("END_ISO must be >= START_ISO")

    async with aiohttp.ClientSession() as session:
        clients = [
            RpcClient(url, session, PER_ENDPOINT_CONCURRENCY) for url in ENDPOINT_URLS
        ]
        # Use first client for bounds
        start_block, end_block = await asyncio.gather(
            find_block_at_or_after_timestamp(clients[0], start_ts),
            find_block_at_or_before_timestamp(clients[0], end_ts),
        )
        if end_block < start_block:
            print("No blocks in given time window", file=sys.stderr)
            return

        # suppress blocks span message to avoid extra output

        # Minimal stats for potential future use; progress printing disabled per user request
        total_blocks = end_block - start_block + 1
        seg_size = max(1, total_blocks // PROGRESS_SEGMENTS)
        fetch_done: List[bool] = [False] * PROGRESS_SEGMENTS
        write_done: List[bool] = [False] * PROGRESS_SEGMENTS
        enqueued_rows_per_seg: List[int] = [0] * PROGRESS_SEGMENTS
        written_rows_per_seg: List[int] = [0] * PROGRESS_SEGMENTS
        stats: Dict[str, int] = {
            "total_ranges": 0,
            "completed_ranges": 0,
            "written_rows": 0,
        }
        state_lock = threading.Lock()

        base_params = build_filter(
            ADDRESSES, TOPIC0, start_block, start_block
        )  # template

        block_ts_cache: Dict[int, int] = {}
        tx_cache: Dict[str, Tuple[Optional[str], Optional[int], Optional[int]]] = {}

        augment_sem = asyncio.Semaphore(AUGMENT_CONCURRENCY)

        # Schedule per-subrange getLogs tasks massively in parallel
        async def process_range(i: int, r: Tuple[int, int]) -> None:
            start, end = r
            client = clients[i % len(clients)]
            try:
                logs = await fetch_logs_adaptive(client, base_params, start, end)
            except Exception as exc:
                print(f"getLogs failed for range {r}: {exc}", file=sys.stderr)
                return
            if not logs:
                return
            if DEBUG:
                if ADDRESSES:
                    addr_str = f"<list:{len(ADDRESSES)}>"
                else:
                    addr_str = "<any>"
                print(
                    f"Fetched {len(logs)} logs in range [{start}, {end}] for address={addr_str}"
                )
            await augment_and_enqueue(
                logs, clients, write_q, augment_sem, block_ts_cache, tx_cache
            )

        ranges = list(chunked_ranges(start_block, end_block, MAX_BLOCKS_PER_QUERY))
        with state_lock:
            stats["total_ranges"] = len(ranges)

        # Start writer thread
        writer_thr = threading.Thread(
            target=writer_thread_fn,
            args=(
                write_q,
                addr_suffix,
                header,
                CSV_BATCH_SIZE,
                DEFER_WRITE,
                start_block,
                seg_size,
                write_done,
                written_rows_per_seg,
                enqueued_rows_per_seg,
                stats,
                state_lock,
                fetch_done,
            ),
            daemon=True,
        )
        writer_thr.start()

        await asyncio.gather(*[process_range(i, r) for i, r in enumerate(ranges)])

    # stop writer
    write_q.put(None)
    writer_thr.join()
    # Suppress final writer/progress messages; keep console focused on fetched log lines


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()
