"""
6-step survey pipeline — each worker enqueues the next on success.

Flow:
  fetch_data_task → vectorize_task → clustering_task → analysis_task → report_task → summary_task
"""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from openai import AsyncOpenAI
from rq import Retry

from core.config import get_settings, ClusteringSettings
from core.exceptions import (
    ClusteringError, EmbeddingError, FileMakerError, AnalysisError,
)
from core.logging import configure_logging, get_logger
from models.survey import SurveyResponse, QuestionType
from services import storage

logger = get_logger(__name__)


def _record_stage_duration(conn, job_id: str, stage_name: str, started_at: float) -> None:
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    storage.record_metric(conn, job_id, f"{stage_name}_duration_ms", duration_ms)
    logger.info("stage_duration", job_id=job_id, stage=stage_name, duration_ms=duration_ms)


def _guard_stage_start(conn, job_id: str, stage_name: str, started_at: float, max_runtime_seconds: int) -> bool:
    if storage.is_stage_completed(conn, job_id, stage_name):
        logger.info("stage_already_completed_skip", job_id=job_id, stage=stage_name)
        return False
    existing_started = conn.hget(f"job:{job_id}:metrics", f"{stage_name}_started_at")
    if existing_started:
        try:
            age = time.time() - float(existing_started)
            if age > max_runtime_seconds:
                # Allow recovered/retried jobs to restart stale stages rather than hard-failing forever.
                logger.warning(
                    "stage_runtime_reset",
                    job_id=job_id,
                    stage=stage_name,
                    previous_age_seconds=int(age),
                    max_runtime_seconds=max_runtime_seconds,
                )
        except ValueError:
            pass
    storage.record_metric(conn, job_id, f"{stage_name}_started_at", time.time())
    return True


def _mark_stage_complete(conn, job_id: str, stage_name: str, started_at: float) -> None:
    storage.mark_stage_completed(conn, job_id, stage_name)
    _record_stage_duration(conn, job_id, stage_name, started_at)


def _enqueue_next(q, task, settings, *args) -> None:
    q.enqueue(
        task,
        *args,
        job_timeout=settings.redis.job_timeout,
        retry=Retry(max=settings.redis.job_retry_max, interval=settings.redis.retry_backoff_seconds),
    )


# ─────────────────────────────────────────────────────────────────────
# WORKER 1 — Fetch Data
# ─────────────────────────────────────────────────────────────────────

def fetch_data_task(job_id: str, event_id: str, record_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    from queues.connection import get_redis_connection, get_question_queue
    from services.filemaker import FileMakerClient

    conn = get_redis_connection()
    started_at = time.perf_counter()
    if not _guard_stage_start(
        conn, job_id, "fetch", started_at, settings.redis.stage_max_runtime_seconds
    ):
        return
    queued_at = conn.hget(f"job:{job_id}:metrics", "queued_at")
    if queued_at:
        from datetime import datetime, timezone
        try:
            queued_dt = datetime.fromisoformat(queued_at.decode() if isinstance(queued_at, bytes) else queued_at)
            queue_wait_ms = int((datetime.now(timezone.utc) - queued_dt).total_seconds() * 1000)
            storage.record_metric(conn, job_id, "queue_wait_ms", max(queue_wait_ms, 0))
            logger.info("queue_wait_recorded", job_id=job_id, queue_wait_ms=max(queue_wait_ms, 0))
        except Exception:
            logger.warning("queue_wait_parse_failed", job_id=job_id)

    logger.info("=== WORKER 1: FETCH DATA — START ===",
                job_id=job_id, event_id=event_id, record_id=record_id)

    storage.update_state(conn, job_id, "fetching", 0.05, "Fetching survey responses from FileMaker")

    try:
        async def _fetch():
            async with FileMakerClient(settings.filemaker) as fm:
                return await fm.fetch_event_responses(event_id)

        data = asyncio.run(_fetch())

        if not data:
            logger.error("fetch_no_data", job_id=job_id, event_id=event_id)
            storage.update_state(conn, job_id, "failed", 0.0,
                error=f"No data returned for event_id={event_id}")
            return

        text_qs    = data.get("text_questions", [])
        numeric_qs = data.get("numeric_questions", {})

        # Log structure of fetched data
        logger.info("fetch_data_keys", job_id=job_id, top_level_keys=list(data.keys()))
        logger.info("fetch_survey_meta",
                    job_id=job_id,
                    survey_id=data.get("survey_id", "—"),
                    survey_name=data.get("survey_name", "—"),
                    record_id=data.get("record_id", "—"),
                    survey_recipients=data.get("survey_recipients", "—"))

        logger.info("fetch_text_questions_count", job_id=job_id, count=len(text_qs))
        for q in text_qs:
            resp_count = len(q.get("responses", []))
            non_empty  = sum(1 for r in q.get("responses", []) if r.get("answer", "").strip())
            logger.info("fetch_text_question",
                        job_id=job_id,
                        question_id=q.get("question_id"),
                        question_text=q.get("question_text", "")[:80],
                        total_responses=resp_count,
                        non_empty_responses=non_empty)

        logger.info("fetch_numeric_questions_count", job_id=job_id, count=len(numeric_qs))
        for qid, q in numeric_qs.items():
            exp_count = len([e for e in q.get("explanation", []) if e.strip()])
            logger.info("fetch_numeric_question",
                        job_id=job_id,
                        question_id=qid,
                        question_text=q.get("question_text", "")[:80],
                        question_type=q.get("question_type", "—"),
                        options=list(q.get("options", {}).keys()),
                        explanation_count=exp_count)

        logger.info("=== WORKER 1: FETCH DATA — COMPLETE ===",
                    job_id=job_id,
                    open_questions=len(text_qs),
                    numeric_questions=len(numeric_qs))

        data["event_id"] = event_id
        storage.store_data(conn, job_id, data)
        storage.update_state(conn, job_id, "vectorizing", 0.10,
            f"Fetched {len(text_qs)} open + {len(numeric_qs)} numeric questions")

        q = get_question_queue()
        _mark_stage_complete(conn, job_id, "fetch", started_at)
        _enqueue_next(q, vectorize_task, settings, job_id)

    except FileMakerError as exc:
        logger.error("fetch_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "fetch_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"FileMaker error: {exc}")
    except Exception as exc:
        logger.error("fetch_unexpected_error", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "fetch_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Unexpected error during fetch: {exc}")


# ─────────────────────────────────────────────────────────────────────
# WORKER 2 — Vectorize
# ─────────────────────────────────────────────────────────────────────

def vectorize_task(job_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    from queues.connection import get_redis_connection, get_question_queue
    from services.embedding import EmbeddingService

    conn = get_redis_connection()
    started_at = time.perf_counter()
    if not _guard_stage_start(
        conn, job_id, "vectorize", started_at, settings.redis.stage_max_runtime_seconds
    ):
        return

    logger.info("=== WORKER 2: VECTORIZE — START ===", job_id=job_id)

    storage.update_state(conn, job_id, "vectorizing", 0.15, "Building response groups and vectorizing")

    try:
        data       = storage.load_data(conn, job_id)
        text_qs    = data.get("text_questions", [])
        numeric_qs = data.get("numeric_questions", {})

        logger.info("vectorize_input",
                    job_id=job_id,
                    text_questions=len(text_qs),
                    numeric_questions=len(numeric_qs))

        # Build groups: open-ended text questions + explanation texts from numeric questions.
        # groups = { gid: (question_text, [SurveyResponse]) }
        groups: dict[str, tuple[str, list[SurveyResponse]]] = {}

        for q in text_qs:
            qid   = q["question_id"]
            qtext = q.get("question_text", "")
            resps = []
            for i, r in enumerate(q.get("responses", [])):
                answer = r.get("answer", "").strip()
                if answer:
                    resps.append(SurveyResponse(
                        record_id=r.get("record_id", f"{qid}_{i}"),
                        respondent_id=r.get("respondent_id", str(i)),
                        question_id=qid,
                        question_text=qtext,
                        question_type=QuestionType.OPEN_ENDED,
                        response_text=answer,
                    ))
            if resps:
                groups[qid] = (qtext, resps)
                logger.info("vectorize_group_built",
                            job_id=job_id, group_id=qid,
                            question_text=qtext[:80],
                            response_count=len(resps),
                            sample_responses=[r.response_text[:60] for r in resps[:3]])

        for qid, q in numeric_qs.items():
            gid   = f"{qid}_exp"
            qtext = q.get("question_text", f"Explanation for {qid}")
            resps = [
                SurveyResponse(
                    record_id=f"{gid}_{i}",
                    respondent_id=str(i),
                    question_id=gid,
                    question_text=qtext,
                    question_type=QuestionType.OPEN_ENDED,
                    response_text=text.strip(),
                )
                for i, text in enumerate(q.get("explanation", []))
                if text.strip()
            ]
            if resps:
                groups[gid] = (qtext, resps)
                logger.info("vectorize_explanation_group_built",
                            job_id=job_id, group_id=gid,
                            question_text=qtext[:80],
                            explanation_count=len(resps),
                            sample_explanations=[r.response_text[:60] for r in resps[:3]])

        if not groups:
            logger.info("vectorize_no_groups", job_id=job_id, message="No groups to vectorize — skipping")
            storage.update_state(conn, job_id, "clustering", 0.40,
                "No response groups to vectorize")
            q = get_question_queue()
            _mark_stage_complete(conn, job_id, "vectorize", started_at)
            _enqueue_next(q, clustering_task, settings, job_id)
            return

        logger.info("vectorize_groups_summary",
                    job_id=job_id,
                    total_groups=len(groups),
                    group_ids=list(groups.keys()),
                    responses_per_group={gid: len(resps) for gid, (_, resps) in groups.items()})

        embedding_service = EmbeddingService(settings.openai)

        async def _embed_all() -> dict[str, Any]:
            semaphore = asyncio.Semaphore(5)

            async def _embed_one(gid: str, resps) -> tuple[str, Any]:
                texts = [r.response_text for r in resps]
                logger.info("vectorize_embedding_start",
                            job_id=job_id, group_id=gid,
                            texts_count=len(texts),
                            batch_size=settings.openai.embedding_batch_size)
                async with semaphore:
                    emb = await embedding_service.embed_texts(texts)
                logger.info("vectorize_embedding_done",
                            job_id=job_id, group_id=gid,
                            vectors_shape=list(emb.shape) if hasattr(emb, "shape") else len(emb),
                            embedding_dim=emb.shape[1] if hasattr(emb, "shape") and len(emb.shape) > 1 else "—")
                return gid, emb

            pairs = await asyncio.gather(*[
                _embed_one(gid, resps) for gid, (_, resps) in groups.items()
            ])
            return dict(pairs)

        vectors = asyncio.run(_embed_all())

        total_vectors = sum(v.shape[0] for v in vectors.values())
        logger.info("=== WORKER 2: VECTORIZE — COMPLETE ===",
                    job_id=job_id,
                    total_groups=len(vectors),
                    total_vectors=total_vectors,
                    vectors_per_group={gid: v.shape[0] for gid, v in vectors.items()})

        storage.update_state(conn, job_id, "vectorizing", 0.38,
            f"Vectorized {len(groups)} groups in parallel")
        storage.store_vectors(conn, job_id, vectors)

        # Serialize groups for clustering worker: {gid: [qtext, [resp_dicts]]}
        groups_serial = {
            gid: [qtext, [r.model_dump() for r in resps]]
            for gid, (qtext, resps) in groups.items()
        }
        storage.store_responses(conn, job_id, groups_serial)

        storage.update_state(conn, job_id, "clustering", 0.40,
            f"Vectorization complete — {total_vectors} total embeddings")

        q = get_question_queue()
        _mark_stage_complete(conn, job_id, "vectorize", started_at)
        _enqueue_next(q, clustering_task, settings, job_id)

    except EmbeddingError as exc:
        logger.error("vectorize_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "vectorize_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Embedding error: {exc}")
    except Exception as exc:
        logger.error("vectorize_unexpected_error", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "vectorize_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Unexpected error during vectorization: {exc}")


# ─────────────────────────────────────────────────────────────────────
# WORKER 3 — Clustering
# ─────────────────────────────────────────────────────────────────────

def clustering_task(job_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    from queues.connection import get_redis_connection, get_question_queue
    from services.clustering import ClusteringService

    conn = get_redis_connection()
    started_at = time.perf_counter()
    if not _guard_stage_start(
        conn, job_id, "clustering", started_at, settings.redis.stage_max_runtime_seconds
    ):
        return

    logger.info("=== WORKER 3: CLUSTERING — START ===", job_id=job_id)

    storage.update_state(conn, job_id, "clustering", 0.42, "Clustering responses by group")

    try:
        groups_serial = storage.load_responses(conn, job_id)  # {gid: [qtext, [resp_dicts]]}
        vectors       = storage.load_vectors(conn, job_id)

        logger.info("clustering_input",
                    job_id=job_id,
                    groups=list(groups_serial.keys()) if groups_serial else [],
                    vectors_loaded=list(vectors.keys()) if vectors else [])

        if not groups_serial:
            logger.info("clustering_no_groups", job_id=job_id, message="No groups — skipping clustering")
            storage.update_state(conn, job_id, "analyzing", 0.60, "No groups — skipping clustering")
            q = get_question_queue()
            _mark_stage_complete(conn, job_id, "clustering", started_at)
            _enqueue_next(q, analysis_task, settings, job_id)
            return

        cluster_results: list[dict] = []
        total = len(groups_serial)
        max_workers = min(4, max(1, total))
        storage.record_metric(conn, job_id, "clustering_parallel_workers", max_workers)

        def _cluster_one(gid: str, group_val: list[Any]) -> dict[str, Any]:
            qtext, resp_dicts = group_val
            resps = [SurveyResponse(**r) for r in resp_dicts]
            embeddings = vectors[gid]
            n = len(resps)
            min_cs = max(2, min(settings.clustering.min_cluster_size, n // 3))
            min_s = max(1, min(settings.clustering.min_samples, min_cs - 1))
            svc = ClusteringService(ClusteringSettings(
                min_cluster_size=min_cs,
                min_samples=min_s,
                metric=settings.clustering.metric,
                cluster_selection_method=settings.clustering.cluster_selection_method,
            ))
            result = svc.cluster_question(gid, resps, embeddings)
            return {"group_id": gid, "question_text": qtext, "result": result.model_dump()}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_cluster_one, gid, group_val) for gid, group_val in groups_serial.items()]
            for idx, future in enumerate(futures):
                clustered = future.result()
                cluster_results.append(clustered)
                progress = 0.42 + 0.18 * ((idx + 1) / total)
                storage.update_state(conn, job_id, "clustering", progress, f"Clustered {idx + 1}/{total} groups")

        logger.info("=== WORKER 3: CLUSTERING — COMPLETE ===",
                    job_id=job_id,
                    groups_processed=len(cluster_results),
                    summary=[{
                        "group_id": r["group_id"],
                        "clusters": len(r["result"].get("clusters", [])),
                        "noise": r["result"].get("noise_count", 0),
                    } for r in cluster_results])

        storage.store_clusters(conn, job_id, cluster_results)
        storage.update_state(conn, job_id, "analyzing", 0.60,
            f"Clustering complete — {len(cluster_results)} groups processed")

        q = get_question_queue()
        _mark_stage_complete(conn, job_id, "clustering", started_at)
        _enqueue_next(q, analysis_task, settings, job_id)

    except ClusteringError as exc:
        logger.error("clustering_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "clustering_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Clustering error: {exc}")
    except Exception as exc:
        logger.error("clustering_unexpected_error", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "clustering_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Unexpected error during clustering: {exc}")


# ─────────────────────────────────────────────────────────────────────
# WORKER 4 — Analysis
# ─────────────────────────────────────────────────────────────────────

def analysis_task(job_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    from queues.connection import get_redis_connection, get_question_queue
    from services.analysis import AnalysisService

    conn = get_redis_connection()
    started_at = time.perf_counter()
    if not _guard_stage_start(
        conn, job_id, "analysis", started_at, settings.redis.stage_max_runtime_seconds
    ):
        return

    logger.info("=== WORKER 4: ANALYSIS — START ===", job_id=job_id)

    storage.update_state(conn, job_id, "analyzing", 0.62, "Running AI analysis on all questions")

    try:
        data          = storage.load_data(conn, job_id)
        groups_serial = storage.load_responses(conn, job_id)  # {gid: [qtext, [resp_dicts]]}
        text_qs       = data.get("text_questions", [])
        numeric_qs    = data.get("numeric_questions", {})

        logger.info("analysis_input",
                    job_id=job_id,
                    open_question_groups=list(groups_serial.keys()) if groups_serial else [],
                    numeric_question_ids=list(numeric_qs.keys()))

        open_qids = {q["question_id"] for q in text_qs}

        # questions_for_analysis: { qid: (qtext, [response_texts], numeric_ctx | None) }
        questions_for_analysis: dict[str, tuple[str, list[str], dict | None]] = {}

        # Open-ended questions — no numeric context
        for gid, group_val in groups_serial.items():
            qtext, resp_dicts = group_val
            if gid in open_qids:
                texts = [r["response_text"] for r in resp_dicts
                         if r.get("response_text", "").strip()]
                questions_for_analysis[gid] = (qtext, texts, None)
                logger.info("analysis_open_question",
                            job_id=job_id,
                            question_id=gid,
                            question_text=qtext[:80],
                            response_count=len(texts),
                            sample_responses=[t[:60] for t in texts[:3]])

        # Numeric questions — merge option aggregates + explanation texts
        for qid, q in numeric_qs.items():
            exp_texts   = [e.strip() for e in q.get("explanation", []) if e.strip()]
            numeric_ctx = {
                "question_type": q.get("question_type", "structured"),
                "options":       q.get("options", {}),
                "total_answers": q.get("total_answers", 0),
            }
            questions_for_analysis[qid] = (
                q.get("question_text", qid),
                exp_texts,
                numeric_ctx,
            )
            logger.info("analysis_numeric_question",
                        job_id=job_id,
                        question_id=qid,
                        question_text=q.get("question_text", qid)[:80],
                        question_type=numeric_ctx["question_type"],
                        options=list(numeric_ctx["options"].keys()),
                        total_answers=numeric_ctx["total_answers"],
                        explanation_count=len(exp_texts))

        if not questions_for_analysis:
            logger.info("analysis_no_questions", job_id=job_id)
            storage.update_state(conn, job_id, "generating_report", 0.80,
                "No questions to analyse — skipping")
            q = get_question_queue()
            _mark_stage_complete(conn, job_id, "analysis", started_at)
            _enqueue_next(q, report_task, settings, job_id)
            return

        logger.info("analysis_questions_summary",
                    job_id=job_id,
                    total_questions=len(questions_for_analysis),
                    question_ids=list(questions_for_analysis.keys()),
                    model=settings.openai.completion_model)

        analysis_service = AnalysisService(settings.openai)
        analyses = asyncio.run(
            analysis_service.analyze_all_questions(questions_for_analysis)
        )

        # Log each analysis result
        for a in analyses:
            s = a.get("summary", a)
            logger.info("analysis_result",
                        job_id=job_id,
                        question_id=a.get("question_id"),
                        question_text=s.get("question_text", "")[:80],
                        question_type=s.get("question_type", "—"),
                        sentiment=s.get("sentiment", "—"),
                        total_responses=s.get("total_responses", "—"),
                        majority_view=s.get("majority_view", "—")[:100],
                        key_findings_count=len(s.get("key_findings", [])),
                        key_findings=[f[:80] for f in s.get("key_findings", [])[:3]])

        logger.info("=== WORKER 4: ANALYSIS — COMPLETE ===",
                    job_id=job_id,
                    questions_analysed=len(analyses),
                    question_ids=[a.get("question_id") for a in analyses])

        storage.store_analysis(conn, job_id, analyses)
        storage.update_state(conn, job_id, "generating_report", 0.80,
            f"Analysis complete — {len(analyses)} questions analysed")

        q = get_question_queue()
        _mark_stage_complete(conn, job_id, "analysis", started_at)
        _enqueue_next(q, report_task, settings, job_id)

    except AnalysisError as exc:
        logger.error("analysis_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "analysis_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Analysis error: {exc}")
    except Exception as exc:
        logger.error("analysis_unexpected_error", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "analysis_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Unexpected error during analysis: {exc}")


# ─────────────────────────────────────────────────────────────────────
# WORKER 5 — Report Generation + Store
# ─────────────────────────────────────────────────────────────────────

def report_task(job_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    from queues.connection import get_redis_connection
    from services.filemaker import FileMakerClient
    from services.analysis import AnalysisService, merge_report_outputs

    conn = get_redis_connection()
    started_at = time.perf_counter()
    if not _guard_stage_start(
        conn, job_id, "report", started_at, settings.redis.stage_max_runtime_seconds
    ):
        return

    logger.info("=== WORKER 5: REPORT GENERATION — START ===", job_id=job_id)

    storage.update_state(conn, job_id, "generating_report", 0.82,
        "Running Report Writer and Risk agents in parallel")

    try:
        data     = storage.load_data(conn, job_id)
        analyses = storage.load_analysis(conn, job_id)

        # Build meta in the full format the report writer expects
        raw_meta = data.get("meta", {})
        try:
            total_respondents = int(data.get("survey_recipients", raw_meta.get("survey_recipients", 0)) or 0)
        except (ValueError, TypeError):
            total_respondents = 0

        survey_meta = {
           "survey_id":         data.get("survey_id",          raw_meta.get("survey_id", "")),
            "event_id":          data.get("event_id",           raw_meta.get("event_id", "")),
            "survey_name":       data.get("survey_name",        raw_meta.get("survey_name", "")),
            "survey_overview":   data.get("survey_description", raw_meta.get("survey_overview", "")),
            "survey_recipients": data.get("survey_recipients",  raw_meta.get("survey_recipients", "0")),
            "version":           raw_meta.get("version", ""),
            "createdBy":         raw_meta.get("createdBy", ""),
            "creationDate":      raw_meta.get("creationDate", ""),
            "deadlineDate":      raw_meta.get("deadlineDate", ""),
            "language":          raw_meta.get("language", "en"),
            "total_respondents": total_respondents,
            "record_id":         data.get("record_id", ""),
            "industry":          data.get("industry", ""),
        }

        logger.info("report_input_meta", job_id=job_id, survey_meta=survey_meta)
        logger.info("report_input_analyses",
                    job_id=job_id,
                    analyses_count=len(analyses),
                    question_ids=[a.get("question_id") for a in analyses],
                    model=settings.openai.completion_model)

        analysis_service = AnalysisService(settings.openai)

        # Run risk agent first so its recommendations feed into the report writer input
        logger.info("report_risk_agent_start", job_id=job_id,
                    message="Running Risk + Action Plan agent first")
        async def _run_risk():
            return await analysis_service.generate_risk_and_action_plan(
                analyses, raw_data=data, meta=survey_meta
            )

        risk_out = asyncio.run(_run_risk())

        logger.info("risk_agent_done",
                    job_id=job_id,
                    output_keys=list(risk_out.keys()) if isinstance(risk_out, dict) else "raw_string",
                    actions_count=len(
                        risk_out.get("recommendations", {})
                            .get("recommendationsWithPlan", {})
                            .get("actions", [])
                    ) if isinstance(risk_out, dict) else "—",
                    positive_state="positiveFeedbacks" in risk_out if isinstance(risk_out, dict) else False)

        # Now run report writer with recommendations embedded in the input
        recommendations = risk_out.get("recommendations", risk_out) if isinstance(risk_out, dict) else {}

        logger.info("report_writer_start", job_id=job_id,
                    message="Running Report Writer agent with recommendations embedded in input")
        async def _run_report():
            return await analysis_service.generate_report(
                analyses,
                survey_meta,
                raw_data=data,
                recommendations=recommendations,
            )

        report_out = asyncio.run(_run_report())

        logger.info("report_writer_output",
                    job_id=job_id,
                    output_keys=list(report_out.keys()) if isinstance(report_out, dict) else "raw_string",
                    questions_in_report=len(report_out.get("questions", [])) if isinstance(report_out, dict) else "—",
                    report_title=report_out.get("reportTitle", "—") if isinstance(report_out, dict) else "—",
                    sentiment_score=report_out.get("sentimentScore", "—") if isinstance(report_out, dict) else "—")

        merged, _ = merge_report_outputs(report_out, risk_out)

        logger.info("report_merged",
                    job_id=job_id,
                    merged_top_level_keys=list(merged.keys()),
                    questions_count=len(merged.get("questions", [])))

        target_lang = merged.get("meta", {}).get("language", "en")
        if target_lang and target_lang.lower() not in ["en", "english", ""]:
            logger.info("report_translation_start", job_id=job_id, target_lang=target_lang)
            merged = analysis_service.translate_report_content(merged, target_lang)
            logger.info("report_translation_done", job_id=job_id)

        fm_record_id = data.get("record_id", "")
        if not fm_record_id:
            logger.error("report_no_record_id", job_id=job_id)
            storage.update_state(conn, job_id, "failed", 0.0,
                error="No record_id in fetched data — cannot update FileMaker record")
            return

        storage.update_state(conn, job_id, "storing", 0.95, "Storing report to FileMaker")
        report_payload = json.dumps(merged, ensure_ascii=False)

        logger.info("report_storing",
                    job_id=job_id,
                    fm_record_id=fm_record_id,
                    payload_size_chars=len(report_payload),
                    payload_size_kb=round(len(report_payload) / 1024, 1))

        async def _store():
            async with FileMakerClient(settings.filemaker) as fm:
                await fm.update_record(
                    layout="N8N_SURVEY_EVENTS",
                    record_id=fm_record_id,
                    field_data={"asJSON_Report": report_payload},
                )

        asyncio.run(_store())

        logger.info("=== WORKER 5: REPORT GENERATION — COMPLETE ===",
                    job_id=job_id,
                    fm_record_id=fm_record_id,
                    payload_size_kb=round(len(report_payload) / 1024, 1),
                    status="stored_to_filemaker")

        storage.cleanup_temp_data(conn, job_id)

        event_id = data.get("event_id", "")
        storage.update_state(conn, job_id, "summarizing", 1.0,
            "Report stored — generating summary")

        from queues.connection import get_question_queue
        q = get_question_queue()
        _mark_stage_complete(conn, job_id, "report", started_at)
        _enqueue_next(q, summary_task, settings, job_id, event_id, fm_record_id)

    except AnalysisError as exc:
        logger.error("report_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "report_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Report generation error: {exc}")
    except FileMakerError as exc:
        logger.error("report_store_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "report_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Failed to store report to FileMaker: {exc}")
    except Exception as exc:
        logger.error("report_unexpected_error", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "report_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Unexpected error during report generation: {exc}")


# ─────────────────────────────────────────────────────────────────────
# WORKER 6 — Summary (Executive Summary Agent)
# ─────────────────────────────────────────────────────────────────────


def summary_task(job_id: str, event_id: str, record_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    from queues.connection import get_redis_connection
    from services.filemaker import FileMakerClient
    from services.analysis import AnalysisService

    conn = get_redis_connection()
    started_at = time.perf_counter()
    if not _guard_stage_start(
        conn, job_id, "summary", started_at, settings.redis.stage_max_runtime_seconds
    ):
        return

    logger.info("=== WORKER 6: SUMMARY — START ===",
                job_id=job_id, event_id=event_id, record_id=record_id)

    storage.update_state(conn, job_id, "summarizing", 0.98,
        "Fetching report and generating executive summary")

    try:
        async def _run():
            async with FileMakerClient(settings.filemaker) as fm:
                # 1. Fetch record from FM using ID_Event and ensure exact record match.
                logger.info("summary_fetching_record", job_id=job_id, event_id=event_id)
                records = await fm.find_records(
                    layout="N8N_SURVEY_EVENTS",
                    query=[{"ID_Event": f"=={event_id}"}],
                )
                if not records:
                    raise ValueError(f"No FM record found for event_id={event_id}")
                matching = next((r for r in records if str(r.get("recordId", "")) == str(record_id)), None)
                if not matching:
                    raise ValueError(
                        f"Record mismatch: event_id={event_id} did not return expected record_id={record_id}"
                    )
                field_data = matching.get("fieldData", {})
                as_json_report = field_data.get("asJSON_Report", "")
                if not as_json_report:
                    raise ValueError("asJSON_Report is empty in FM record")

                logger.info("summary_report_fetched",
                            job_id=job_id,
                            report_size=len(as_json_report))

                # 2. Call executive summary agent via AnalysisService
                logger.info("summary_agent_start", job_id=job_id,
                            model=settings.openai.completion_model)
                analysis_service = AnalysisService(settings.openai)
                summary_output = await analysis_service.generate_executive_summary(as_json_report)

                logger.info("summary_agent_done",
                            job_id=job_id,
                            summary_size=len(summary_output))

                # 2.5 Translate if target language is not English
                try:
                    report_obj = json.loads(as_json_report)
                    target_lang = report_obj.get("meta", {}).get("language")
                    
                    logger.info("summary_translation_target_lang",
                                job_id=job_id, target_lang=target_lang)
                    
                    if target_lang and target_lang.lower() not in ["en", "english", ""]:
                        logger.info("summary_translation_start", 
                                    job_id=job_id, target_lang=target_lang)
                        
                        summary_dict = json.loads(summary_output)
                        translated_dict = analysis_service.translate_summary_content(
                            summary_dict, target_lang
                        )
                        summary_output = json.dumps(translated_dict, ensure_ascii=False)
                        
                        logger.info("summary_translation_done", job_id=job_id)
                except Exception as t_exc:
                    logger.warning("summary_translation_skipped", 
                                   job_id=job_id, error=str(t_exc))

                logger.info("summary_output_pre_store",
                            job_id=job_id, record_id=record_id, summary=summary_output)

                # 3. Write asJSON_Summary back to FM
                logger.info("summary_storing", job_id=job_id, record_id=record_id)
                await fm.update_record(
                    layout="N8N_SURVEY_EVENTS",
                    record_id=record_id,
                    field_data={"asJSON_Summary": summary_output},
                )

                # 4. Build and store ConcatReport from both JSONs
                logger.info("summary_concat_start", job_id=job_id, record_id=record_id)
                from services.analysis import merge_concat_report
                concat_report = merge_concat_report(as_json_report, summary_output)
                concat_payload = json.dumps(concat_report, ensure_ascii=False)
                await fm.update_record(
                    layout="N8N_SURVEY_EVENTS",
                    record_id=record_id,
                    field_data={"asJSON_ConcatReport": concat_payload},
                )
                logger.info("summary_concat_stored",
                            job_id=job_id,
                            record_id=record_id,
                            concat_size=len(concat_payload))

        asyncio.run(_run())

        logger.info("=== WORKER 6: SUMMARY — COMPLETE ===",
                    job_id=job_id, event_id=event_id, record_id=record_id)

        storage.update_state(conn, job_id, "complete", 1.0,
            "Summary generated and stored to FileMaker")
        _mark_stage_complete(conn, job_id, "summary", started_at)
        logger.info("pipeline_complete", job_id=job_id, record_id=record_id)

    except ValueError as exc:
        logger.error("summary_validation_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "summary_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Summary validation error: {exc}")
    except FileMakerError as exc:
        logger.error("summary_filemaker_failed", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "summary_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Summary FileMaker error: {exc}")
    except Exception as exc:
        logger.error("summary_unexpected_error", job_id=job_id, error=str(exc))
        storage.incr_metric(conn, job_id, "summary_failures")
        storage.update_state(conn, job_id, "failed", 0.0,
            error=f"Unexpected error during summary generation: {exc}")
