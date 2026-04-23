import argparse
import time
import threading
from datetime import datetime, timezone
from rq import SimpleWorker, Queue
from rq.timeouts import BaseDeathPenalty
from rq import Retry
from queues.connection import get_redis_connection
from core.config import get_settings
from core.logging import configure_logging, get_logger

logger = get_logger(__name__)


class NoopDeathPenalty(BaseDeathPenalty):
    """No-op timeout enforcement — required on Windows (no SIGALRM)."""
    def setup_death_penalty(self):
        pass

    def teardown_death_penalty(self):
        pass

    def cancel_death_penalty(self):
        pass


class WindowsWorker(SimpleWorker):
    death_penalty_class = NoopDeathPenalty


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _recover_stale_jobs(conn, settings) -> int:
    now = datetime.now(timezone.utc)
    stale_cutoff = settings.redis.stale_job_recovery_seconds
    active_statuses = {"queued", "fetching", "vectorizing", "clustering", "analyzing", "generating_report", "storing", "summarizing"}
    recovered = 0

    status_to_task = {
        "queued": ("fetch_data_task", ("event_id", "record_id")),
        "fetching": ("fetch_data_task", ("event_id", "record_id")),
        "vectorizing": ("vectorize_task", ()),
        "clustering": ("clustering_task", ()),
        "analyzing": ("analysis_task", ()),
        "generating_report": ("report_task", ()),
        # "storing" means report persistence was in progress; resume report stage first.
        "storing": ("report_task", ()),
        "summarizing": ("summary_task", ("event_id", "record_id")),
    }

    for key in conn.scan_iter("job:*:state"):
        state = conn.hgetall(key)
        if not state:
            continue
        decoded = {_decode(k): _decode(v) for k, v in state.items()}
        status = decoded.get("status", "")
        if status not in active_statuses:
            continue
        updated_at = decoded.get("updated_at", "")
        if not updated_at:
            continue
        try:
            updated_dt = datetime.fromisoformat(updated_at)
        except ValueError:
            continue
        age_seconds = (now - updated_dt).total_seconds()
        if age_seconds < stale_cutoff:
            continue

        job_id = _decode(key).split(":")[1]
        if conn.get(f"job:{job_id}:recovering"):
            continue

        task_name, arg_fields = status_to_task.get(status, ("fetch_data_task", ("event_id", "record_id")))
        from workers import tasks as pipeline_tasks
        task_fn = getattr(pipeline_tasks, task_name)
        args = [job_id]
        for fld in arg_fields:
            args.append(decoded.get(fld, ""))

        queue = Queue("question_processing", connection=conn)
        queue.enqueue(
            task_fn,
            *args,
            job_timeout=settings.redis.job_timeout,
            retry=Retry(max=settings.redis.job_retry_max, interval=settings.redis.retry_backoff_seconds),
        )
        conn.setex(f"job:{job_id}:recovering", 300, "1")
        conn.hset(key, mapping={
            "status": "queued",
            "message": f"Recovered stale job from status={status}",
            "updated_at": now.isoformat(),
        })
        recovered += 1
        logger.warning("stale_job_recovered", job_id=job_id, previous_status=status, age_seconds=int(age_seconds))
    return recovered


if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Start RQ worker")
    parser.add_argument(
        "--recover-stale-interval-seconds",
        type=int,
        default=30,
        help="Interval to scan and recover stale pipeline jobs.",
    )
    parser.add_argument(
        "--disable-stale-recovery",
        action="store_true",
        help="Disable stale-job recovery loop (useful when multiple workers are running).",
    )
    parser.add_argument(
        "--worker-name",
        default="",
        help="Optional explicit worker name for observability/autoscaler tracking.",
    )
    parser.add_argument(
        "--queues",
        nargs="+",
        default=["survey_pipeline", "question_processing", "default"],
        help="Queue names to listen on",
    )
    args = parser.parse_args()

    conn = get_redis_connection()
    queues = [Queue(name, connection=conn) for name in args.queues]
    worker_name = args.worker_name or None
    worker = WindowsWorker(queues, connection=conn, name=worker_name)
    if not args.disable_stale_recovery:
        logger.info("worker_bootstrap_stale_recovery_start")
        _recover_stale_jobs(conn, settings)
        logger.info("worker_bootstrap_stale_recovery_done")

    def _stale_recovery_loop() -> None:
        while True:
            time.sleep(args.recover_stale_interval_seconds)
            try:
                recovered = _recover_stale_jobs(conn, settings)
                if recovered:
                    logger.info("stale_recovery_batch", recovered=recovered)
            except Exception as exc:
                logger.error("stale_recovery_loop_error", error=str(exc))

    if not args.disable_stale_recovery:
        threading.Thread(target=_stale_recovery_loop, daemon=True).start()

    # Persistent worker process; avoids burst-mode subscribe/unsubscribe loop spam.
    worker.work()
