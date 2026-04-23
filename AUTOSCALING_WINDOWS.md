# Windows Autoscaling Runbook

This project now includes a process autoscaler for worker capacity:

- `scripts/autoscale_workers.py`
and a Docker Compose autoscaler for whole-system scaling (API + workers behind Nginx):

- `docker-compose.autoscale.yml`
- `scripts/docker_autoscaler.py`

It scales worker process count based on Redis queue depth.

## Recommended Start Order

1. Start Redis
2. For whole-system autoscaling (recommended):
   - Start the stack (Nginx ingress + API + workers + Redis):
     - `docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.autoscale.yml up -d`
   - Start autoscaler (scales `api` + `worker-questions`):
     - `python scripts/docker_autoscaler.py --redis-url redis://localhost:6379/0`

If you do NOT want Docker autoscaling, you can run process-based scaling:
- API (fixed size):
  - `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2`
- Worker autoscaler:
  - `python scripts/autoscale_workers.py --queue question_processing --min-workers 1 --max-workers 8 --target-queue-per-worker 5`
- Pipeline worker:
  - `python -m workers.runner --queues survey_pipeline`

## Scaling Logic

- Desired workers:
  - `ceil(queue_depth / target_queue_per_worker)`
  - clamped between `min_workers` and `max_workers`
- Uses cooldown to avoid rapid scale flapping.
- Keeps one primary worker with stale-recovery enabled.
- Additional scaled workers run with `--disable-stale-recovery`.
- Scales down only idle workers (busy workers are not stopped).
- Uses Redis leader lock so only one autoscaler instance is active per queue.

## Suggested Production Values (<=10 req/s target)

- `--min-workers 1`
- `--max-workers 6`
- `--target-queue-per-worker 4`
- `--poll-seconds 10`
- `--scale-cooldown-seconds 30`
- `--lock-ttl-seconds 60`

Tune after observing queue depth and job duration.

## Operational Notes

- If autoscaler exits, existing worker processes may continue running.
- Stop autoscaler with `Ctrl+C`; it will attempt graceful shutdown of managed workers.
- You can run a standby autoscaler process on another terminal/server; leader lock prevents split-brain scaling.
- Monitor:
  - queue depth (`/health/`)
  - job failure metrics
  - Redis memory/evictions
  - webhook overload rate (`503`)
