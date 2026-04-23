import argparse
import asyncio
import json
import statistics
import time
import uuid
from typing import Any

import httpx


def _payload() -> dict[str, str]:
    return {
        "event_id": f"load-{uuid.uuid4()}",
        "record_id": f"rec-{uuid.uuid4()}",
    }


async def _one_request(client: httpx.AsyncClient, url: str) -> tuple[int, float]:
    started = time.perf_counter()
    resp = await client.post(url, json=_payload())
    elapsed_ms = (time.perf_counter() - started) * 1000
    return resp.status_code, elapsed_ms


async def run_load_test(
    base_url: str,
    api_key: str,
    total_requests: int,
    concurrency: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/webhook/"
    timeout = httpx.Timeout(10.0, connect=5.0)
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    status_counts: dict[int, int] = {}

    async with httpx.AsyncClient(
        headers={"x-api-key": api_key, "content-type": "application/json"},
        timeout=timeout,
        limits=limits,
    ) as client:
        async def _run_one() -> None:
            async with sem:
                status, latency = await _one_request(client, url)
                latencies.append(latency)
                status_counts[status] = status_counts.get(status, 0) + 1

        tasks = [_run_one() for _ in range(total_requests)]
        started = time.perf_counter()
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - started

    latencies_sorted = sorted(latencies)
    def pct(p: float) -> float:
        if not latencies_sorted:
            return 0.0
        idx = min(len(latencies_sorted) - 1, int((p / 100.0) * len(latencies_sorted)))
        return latencies_sorted[idx]

    success = sum(v for k, v in status_counts.items() if 200 <= k < 300)
    failures = total_requests - success

    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "duration_seconds": round(total_time, 3),
        "requests_per_second": round(total_requests / total_time, 2) if total_time > 0 else 0.0,
        "status_counts": status_counts,
        "success_rate": round((success / total_requests) * 100, 2) if total_requests else 0.0,
        "failure_rate": round((failures / total_requests) * 100, 2) if total_requests else 0.0,
        "latency_ms": {
            "p50": round(pct(50), 2),
            "p95": round(pct(95), 2),
            "p99": round(pct(99), 2),
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Webhook load test runner")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--total-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    result = asyncio.run(
        run_load_test(
            base_url=args.base_url,
            api_key=args.api_key,
            total_requests=args.total_requests,
            concurrency=args.concurrency,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
