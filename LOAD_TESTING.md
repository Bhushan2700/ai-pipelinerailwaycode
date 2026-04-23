# Load Validation Guide

This project now includes a webhook load harness at `scripts/load_test_webhook.py`.

## Goal

Validate that ingress can sustain expected traffic without excessive latency or failures while queue depth remains controlled.

## Run

```bash
python scripts/load_test_webhook.py --base-url http://localhost:8000 --api-key <WEBHOOK_API_KEY> --total-requests 400 --concurrency 40
```

## Suggested Profiles

- Normal: `--total-requests 200 --concurrency 20`
- Burst: `--total-requests 600 --concurrency 60`
- Stress: `--total-requests 1000 --concurrency 100`

## SLO Checks

- Webhook success rate: `>= 99%`
- Ingress latency: `p95 < 500ms`, `p99 < 1000ms`
- Failure rate (5xx): `< 1%`
- Queue pressure:
  - `question_processing` queue should drain within your accepted batch window
  - no sustained growth over 10+ minutes
- Redis:
  - no frequent evictions
  - stable used memory below hard limit

## Where To Observe

- `GET /health/` now exposes:
  - `queue_depths`
  - `redis_used_memory`
- Job-level pipeline timings are written in Redis under:
  - `job:<job_id>:metrics`
  - includes `queue_wait_ms` and per-stage duration metrics.
