from typing import Optional
from redis import Redis
from rq import Queue
from core.config import get_settings

_redis_conn: Optional[Redis] = None


def get_redis_connection() -> Redis:
    global _redis_conn
    if _redis_conn is None:
        settings = get_settings()
        _redis_conn = Redis.from_url(settings.redis.url)
    return _redis_conn


def get_pipeline_queue() -> Queue:
    return Queue("survey_pipeline", connection=get_redis_connection())


def get_question_queue() -> Queue:
    return Queue("question_processing", connection=get_redis_connection())


def get_default_queue() -> Queue:
    return Queue("default", connection=get_redis_connection())
