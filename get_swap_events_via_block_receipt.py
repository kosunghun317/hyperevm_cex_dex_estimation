#!/usr/bin/env python3
"""
Async pipeline to fetch full block receipts (eth_getBlockReceipts) and extract Swap events.

Notes:
- Uses asyncio + aiohttp with a global token bucket to cap total RPS (default 100).
- For each block, fetches the header (for timestamp/hash) and the full block receipts.
- Filters logs by Swap topic0 and optional address allowlist, then writes per-day CSVs.
- Non-blocking CSV writing via a dedicated writer thread.

Env overrides:
- HYPEREVM_RPC_URLS / HYPEREVM_RPC_URL
- SWAP_EVENTS_START_ISO / SWAP_EVENTS_END_ISO (ISO8601, default 2025-08-01..2025-08-31)
- SWAP_EVENTS_GLOBAL_RPS (default 100)
- SWAP_EVENTS_ENDPOINT_CONCURRENCY (default 8)
- SWAP_EVENTS_BLOCK_WORKERS (default 64)
- SWAP_EVENTS_CSV_BATCH (default 1000)
- SWAP_EVENTS_DEBUG (default 1)

Requires: aiohttp
"""

import asyncio
import csv
import json
import os
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from queue import Queue
from time import monotonic
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
import math

import polars as pl

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
# CONFIG
# =====================

RPC_URLS_ENV = os.environ.get("HYPEREVM_RPC_URLS", "").strip()
ENDPOINT_URLS: List[str] = [
    u.strip() for u in RPC_URLS_ENV.split(",") if u.strip()
] or [os.environ.get("HYPEREVM_RPC_URL") or "http://localhost:8545"]

# If empty, topic-only search across all pools
ADDRESSES: Optional[List[str]] = [
    "0xBd19E19E4b70eB7F248695a42208bc1EdBBFb57D",
    "0x161fB7d6c764f81DAE581E8a4981772750416727",
    "0x337b56d87a6185cd46af3ac2cdf03cbc37070c30",
    "0x56abfaf40f5b7464e9cc8cff1af13863d6914508",
    "0xf40d57783c3359f160d006b9bc7a2e4311fe6a86",
    "0x68f4Bb01D75Edf476195fCA80271ADf5CA71A242",
    "0x3603ffEbB994CC110b4186040CaC3005B2cf4465",
]
ADDRESSES_LOWER: Optional[Set[str]] = (
    set(a.lower() for a in ADDRESSES) if ADDRESSES else None
)

TOPIC0 = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"  # Swap

GLOBAL_RPS = min(100, int(os.environ.get("SWAP_EVENTS_GLOBAL_RPS", "100")))
PER_ENDPOINT_CONCURRENCY = int(os.environ.get("SWAP_EVENTS_ENDPOINT_CONCURRENCY", "16"))
BLOCK_WORKERS = int(os.environ.get("SWAP_EVENTS_BLOCK_WORKERS", "16"))
CSV_BATCH_SIZE = int(os.environ.get("SWAP_EVENTS_CSV_BATCH", "10_000"))

START_ISO = os.environ.get("SWAP_EVENTS_START_ISO", "2025-08-01T00:00:00Z")
END_ISO = os.environ.get("SWAP_EVENTS_END_ISO", "2025-08-31T00:00:00Z")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "event_datasets_final_async")

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEBUG = os.environ.get("SWAP_EVENTS_DEBUG", "1") == "1"


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


def to_hex_block(n: int) -> str:
    return hex(int(n))


def decode_int256(word_hex: str) -> int:
    b = bytes.fromhex(word_hex)
    return int.from_bytes(b, byteorder="big", signed=True)


def decode_swap_data(data_hex: str) -> Tuple[Optional[int], Optional[int]]:
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

    async def eth_getBlockByNumber(self, block_number: int) -> Dict[str, Any]:
        res = await self._rpc(
            "eth_getBlockByNumber", [to_hex_block(block_number), False]
        )
        return res

    async def eth_getBlockReceipts(self, block_hash: str) -> List[Dict[str, Any]]:
        # Most providers expect a single param: blockHash
        res = await self._rpc("eth_getBlockReceipts", [block_hash])
        return res or []


# =====================
# Block range helpers
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
# Writer thread
# =====================


def ensure_writer(
    day_str: str, addr_suffix: str, files: Dict[str, Any], header: List[str]
) -> Any:
    if day_str in files:
        return files[day_str]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(
        OUTPUT_DIR, f"swap_events_blockrcpt_{addr_suffix}_{day_str}.csv"
    )
    f = open(path, "w", encoding="utf-8", newline="")
    # write header once
    f.write(",".join(header) + "\n")
    files[day_str] = f
    return f


def writer_thread_fn(
    write_q: Queue, addr_suffix: str, header: List[str], batch_size: int
) -> None:
    files: Dict[str, Any] = {}
    buffers: Dict[str, List[List[str]]] = {}
    while True:
        item = write_q.get()
        if item is None:
            break
        day_str, row = item
        buf = buffers.setdefault(day_str, [])
        buf.append(row)
        if len(buf) >= batch_size:
            f = ensure_writer(day_str, addr_suffix, files, header)
            try:
                # build DataFrame and append without header
                cols = list(zip(*buf))
                df = pl.DataFrame(
                    {header[i]: list(cols[i]) for i in range(len(header))}
                )
                df.write_csv(f, include_header=False)
            except Exception:
                # fallback to csv module
                w = csv.writer(f)
                w.writerows(buf)
            buf.clear()
    # flush remaining
    for day_str, buf in buffers.items():
        if not buf:
            continue
        f = ensure_writer(day_str, addr_suffix, files, header)
        try:
            cols = list(zip(*buf))
            df = pl.DataFrame({header[i]: list(cols[i]) for i in range(len(header))})
            df.write_csv(f, include_header=False)
        except Exception:
            w = csv.writer(f)
            w.writerows(buf)
        buf.clear()
    for f in files.values():
        try:
            f.close()
        except Exception:
            pass


# =====================
# Main processing
# =====================


async def process_block(
    block_number: int,
    clients: List[RpcClient],
    out_q: Queue,
    addresses_lower: Optional[Set[str]],
) -> None:
    client = clients[block_number % len(clients)]
    try:
        blk = await client.eth_getBlockByNumber(block_number)
    except Exception as exc:
        print(
            f"eth_getBlockByNumber failed for block {block_number}: {exc}",
            file=sys.stderr,
        )
        return
    try:
        ts = int(blk.get("timestamp"), 16)
    except Exception:
        ts = None
    block_hash = blk.get("hash")
    if not isinstance(block_hash, str):
        # cannot continue
        return
    try:
        receipts = await client.eth_getBlockReceipts(block_hash)
    except Exception as exc:
        print(
            f"eth_getBlockReceipts failed for block {block_number}: {exc}",
            file=sys.stderr,
        )
        return

    matched = 0
    day_str = (
        datetime.utcfromtimestamp(ts).strftime("%Y%m%d")
        if ts is not None
        else "unknown"
    )
    for rcpt in receipts:
        to_field = rcpt.get("to")
        tx_to = to_field if isinstance(to_field, str) and to_field else ZERO_ADDRESS
        gu = rcpt.get("gasUsed")
        gas_used: Optional[int] = (
            int(gu, 16)
            if isinstance(gu, str)
            else (int(gu) if gu is not None else None)
        )
        egp = rcpt.get("effectiveGasPrice")
        gas_price: Optional[int] = (
            int(egp, 16)
            if isinstance(egp, str)
            else (int(egp) if egp is not None else None)
        )
        logs = rcpt.get("logs") or []
        for log in logs:
            topics = log.get("topics") or []
            if not topics:
                continue
            try:
                topic0 = topics[0]
            except Exception:
                continue
            if not isinstance(topic0, str) or topic0.lower() != TOPIC0.lower():
                continue
            pool_address = log.get("address")
            if addresses_lower and (
                not isinstance(pool_address, str)
                or pool_address.lower() not in addresses_lower
            ):
                continue
            data_hex = log.get("data")
            amount0, amount1 = decode_swap_data(data_hex)
            row = [
                str(ts if ts is not None else ""),
                str(block_number),
                tx_to,
                str(amount0 if amount0 is not None else ""),
                str(amount1 if amount1 is not None else ""),
                pool_address,
                str(gas_price if gas_price is not None else ""),
                str(gas_used if gas_used is not None else ""),
            ]
            out_q.put((day_str, row))
            matched += 1
    if DEBUG:
        print(f"Block {block_number}: matched {matched} swap logs")


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
    writer_thr = threading.Thread(
        target=writer_thread_fn,
        args=(write_q, addr_suffix, header, CSV_BATCH_SIZE),
        daemon=True,
    )
    writer_thr.start()

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
            write_q.put(None)
            writer_thr.join()
            return

        # Contiguous range assignment to workers
        total_blocks = end_block - start_block + 1
        chunk = (total_blocks + BLOCK_WORKERS - 1) // BLOCK_WORKERS

        async def worker(idx: int) -> None:
            bstart = start_block + idx * chunk
            if bstart > end_block:
                return
            bend = min(end_block, bstart + chunk - 1)
            for bn in range(bstart, bend + 1):
                await process_block(bn, clients, write_q, ADDRESSES_LOWER)

        workers = [asyncio.create_task(worker(i)) for i in range(BLOCK_WORKERS)]
        await asyncio.gather(*workers)

    # stop writer
    write_q.put(None)
    writer_thr.join()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()
