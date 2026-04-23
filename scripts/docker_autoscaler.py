import argparse
import math
import os
import subprocess
import sys
import time

from redis import Redis
from rq import Queue


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _compose_cmd(files: list[str], args: list[str]) -> list[str]:
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", f]
    cmd += args
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Autoscale docker compose services by queue depth")
    p.add_argument("--redis-url", default=os.environ.get("REDIS__URL", "redis://localhost:6379/0"))
    p.add_argument("--queue", default="question_processing")
    p.add_argument("--compose-files", nargs="+", default=["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.autoscale.yml"])

    p.add_argument("--min-api", type=int, default=1)
    p.add_argument("--max-api", type=int, default=4)
    p.add_argument("--target-queue-per-api", type=int, default=20)

    p.add_argument("--min-workers", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--target-queue-per-worker", type=int, default=5)

    p.add_argument("--poll-seconds", type=int, default=10)
    p.add_argument("--cooldown-seconds", type=int, default=30)
    args = p.parse_args()

    r = Redis.from_url(args.redis_url)
    q = Queue(args.queue, connection=r)

    last_scale = 0.0
    last_api = None
    last_workers = None

    print(f"[docker-autoscaler] using compose files: {args.compose_files}")
    while True:
        depth = q.count
        desired_api = min(args.max_api, max(args.min_api, math.ceil(depth / max(1, args.target_queue_per_api))))
        desired_workers = min(args.max_workers, max(args.min_workers, math.ceil(depth / max(1, args.target_queue_per_worker))))

        now = time.time()
        if (desired_api != last_api or desired_workers != last_workers) and (now - last_scale >= args.cooldown_seconds):
            cmd = _compose_cmd(
                args.compose_files,
                [
                    "up",
                    "-d",
                    "--no-recreate",
                    "--scale",
                    f"api={desired_api}",
                    "--scale",
                    f"worker-questions={desired_workers}",
                ],
            )
            print(f"[docker-autoscaler] scale api={desired_api} worker-questions={desired_workers} (queue_depth={depth})")
            _run(cmd)
            last_api = desired_api
            last_workers = desired_workers
            last_scale = now
        else:
            print(f"[docker-autoscaler] queue_depth={depth} api={last_api} workers={last_workers} desired_api={desired_api} desired_workers={desired_workers}")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

