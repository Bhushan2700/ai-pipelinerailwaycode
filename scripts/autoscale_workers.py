import argparse
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is importable when executed as script.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from redis import Redis
from rq import Queue, Worker

from core.config import get_settings


@dataclass
class WorkerProc:
    name: str
    process: subprocess.Popen
    started_at: float


def _build_command(queue_name: str, worker_name: str, disable_stale_recovery: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "workers.runner",
        "--queues",
        queue_name,
        "--worker-name",
        worker_name,
    ]
    if disable_stale_recovery:
        cmd.append("--disable-stale-recovery")
    return cmd


def _spawn_worker(queue_name: str, worker_name: str, disable_stale_recovery: bool) -> WorkerProc:
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        _build_command(queue_name, worker_name, disable_stale_recovery),
        creationflags=creationflags,
    )
    return WorkerProc(name=worker_name, process=proc, started_at=time.time())


def _stop_worker(worker: WorkerProc, shutdown_grace_seconds: int) -> None:
    proc = worker.process
    if proc.poll() is not None:
        return
    try:
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=shutdown_grace_seconds)
    except Exception:
        proc.kill()


def _is_worker_busy(redis_conn: Redis, worker_name: str) -> bool:
    try:
        workers = Worker.all(connection=redis_conn)
        for w in workers:
            if w.name == worker_name:
                return bool(w.get_current_job_id())
    except Exception:
        # If worker introspection fails, default to "busy" to avoid unsafe scale-down.
        return True
    return False


def _acquire_or_renew_lock(
    redis_conn: Redis,
    lock_key: str,
    lock_value: str,
    lock_ttl_seconds: int,
) -> bool:
    got = redis_conn.set(lock_key, lock_value, nx=True, ex=lock_ttl_seconds)
    if got:
        return True
    owner = redis_conn.get(lock_key)
    owner_val = owner.decode() if isinstance(owner, bytes) else str(owner or "")
    if owner_val == lock_value:
        redis_conn.expire(lock_key, lock_ttl_seconds)
        return True
    return False


def run_autoscaler(
    queue_name: str,
    min_workers: int,
    max_workers: int,
    target_queue_per_worker: int,
    poll_seconds: int,
    scale_cooldown_seconds: int,
    shutdown_grace_seconds: int,
    lock_ttl_seconds: int,
) -> None:
    settings = get_settings()
    redis_conn = Redis.from_url(settings.redis.url)
    queue = Queue(queue_name, connection=redis_conn)
    workers: list[WorkerProc] = []
    last_scale_at = 0.0
    lock_key = f"autoscaler:leader:{queue_name}"
    lock_value = f"{sys.platform}:{time.time()}:{os.getpid()}"
    next_id = 1

    print(f"[autoscaler] standby mode for queue={queue_name}")

    try:
        while True:
            is_leader = _acquire_or_renew_lock(redis_conn, lock_key, lock_value, lock_ttl_seconds)
            if not is_leader:
                # Ensure we don't keep managing workers while not leader.
                if workers:
                    print("[autoscaler] leadership lost; stopping managed workers")
                    for w in workers:
                        _stop_worker(w, shutdown_grace_seconds)
                    workers = []
                time.sleep(poll_seconds)
                continue

            if not workers:
                for idx in range(min_workers):
                    worker_name = f"autoscale-{queue_name}-{next_id}"
                    next_id += 1
                    workers.append(
                        _spawn_worker(
                            queue_name,
                            worker_name,
                            disable_stale_recovery=(idx != 0),
                        )
                    )
                print(f"[autoscaler] leader active, started with {len(workers)} workers")

            # Remove exited workers from pool.
            workers = [w for w in workers if w.process.poll() is None]

            depth = queue.count
            desired = min(max_workers, max(min_workers, math.ceil(depth / max(1, target_queue_per_worker))))

            now = time.time()
            if now - last_scale_at >= scale_cooldown_seconds:
                if desired > len(workers):
                    to_add = desired - len(workers)
                    for _ in range(to_add):
                        # New workers do not run stale-recovery loop.
                        worker_name = f"autoscale-{queue_name}-{next_id}"
                        next_id += 1
                        workers.append(_spawn_worker(queue_name, worker_name, disable_stale_recovery=True))
                    print(f"[autoscaler] scale up -> {len(workers)} workers (queue_depth={depth})")
                    last_scale_at = now
                elif desired < len(workers):
                    to_remove = len(workers) - desired
                    # Stop newest workers first.
                    workers_sorted = sorted(workers, key=lambda w: w.started_at, reverse=True)
                    victims: list[WorkerProc] = []
                    for candidate in workers_sorted:
                        if len(victims) >= to_remove:
                            break
                        if not _is_worker_busy(redis_conn, candidate.name):
                            victims.append(candidate)
                    for victim in victims:
                        _stop_worker(victim, shutdown_grace_seconds)
                    workers = [w for w in workers if w.process.poll() is None]
                    print(
                        f"[autoscaler] scale down -> {len(workers)} workers "
                        f"(queue_depth={depth}, stopped={len(victims)}, requested={to_remove})"
                    )
                    last_scale_at = now

            print(f"[autoscaler] queue_depth={depth} workers={len(workers)} desired={desired}")
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("[autoscaler] stopping all workers...")
        for w in workers:
            _stop_worker(w, shutdown_grace_seconds)
        owner = redis_conn.get(lock_key)
        owner_val = owner.decode() if isinstance(owner, bytes) else str(owner or "")
        if owner_val == lock_value:
            redis_conn.delete(lock_key)
        print("[autoscaler] stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoscale Windows worker processes by Redis queue depth")
    parser.add_argument("--queue", default="question_processing", help="Queue name to autoscale")
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--target-queue-per-worker", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--scale-cooldown-seconds", type=int, default=30)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=20)
    parser.add_argument("--lock-ttl-seconds", type=int, default=60)
    args = parser.parse_args()

    run_autoscaler(
        queue_name=args.queue,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        target_queue_per_worker=args.target_queue_per_worker,
        poll_seconds=args.poll_seconds,
        scale_cooldown_seconds=args.scale_cooldown_seconds,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
        lock_ttl_seconds=args.lock_ttl_seconds,
    )


if __name__ == "__main__":
    main()
