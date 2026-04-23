"""
Step 4: Analyze clusters (or noise groups) per question using GPT.

For our small test dataset everything is noise, so this tests the noise
analysis path — which is exactly what will run on sparse questions in prod.
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from core.config import get_settings, ClusteringSettings
from models.clusters import ClusterAnalysis, QuestionClusterResult
from models.survey import SurveyResponse, QuestionType
from services.filemaker import FileMakerClient
from services.embedding import EmbeddingService
from services.clustering import ClusteringService
from services.analysis import AnalysisService

EVENT_ID = "951F9774-45EF-FD41-899A-4C57745021B5"


def build_responses_from_text_question(q: dict) -> tuple[str, list[SurveyResponse]]:
    qid = q["question_id"]
    qtext = q.get("question_text", "")
    responses = []
    for i, r in enumerate(q.get("responses", [])):
        answer = r.get("answer", "").strip()
        if not answer:
            continue
        responses.append(
            SurveyResponse(
                record_id=r.get("record_id", f"{qid}_{i}"),
                respondent_id=r.get("respondent_id", str(i)),
                question_id=qid,
                question_text=qtext,
                question_type=QuestionType.OPEN_ENDED,
                response_text=answer,
            )
        )
    return qid, responses


def build_responses_from_explanations(qid: str, q: dict) -> tuple[str, list[SurveyResponse]]:
    group_id = f"{qid}_exp"
    qtext = q.get("question_text", f"Explanation for {qid}")
    explanations = [e.strip() for e in q.get("explanation", []) if e.strip()]
    return group_id, [
        SurveyResponse(
            record_id=f"{group_id}_{i}",
            respondent_id=str(i),
            question_id=group_id,
            question_text=qtext,
            question_type=QuestionType.OPEN_ENDED,
            response_text=text,
        )
        for i, text in enumerate(explanations)
    ]


async def main():
    settings = get_settings()

    # ── Fetch ─────────────────────────────────────────────────────────────────
    print("Fetching from FileMaker ...")
    async with FileMakerClient(settings.filemaker) as fm:
        data = await fm.fetch_event_responses(EVENT_ID)

    if not data:
        print("No data.")
        return

    # ── Build response groups ─────────────────────────────────────────────────
    groups: dict[str, list[SurveyResponse]] = {}
    for q in data.get("text_questions", []):
        qid, resps = build_responses_from_text_question(q)
        if resps:
            groups[qid] = resps
    for qid, q in data.get("numeric_questions", {}).items():
        gid, resps = build_responses_from_explanations(qid, q)
        if resps:
            groups[gid] = resps

    # ── Vectorize ─────────────────────────────────────────────────────────────
    print("Vectorizing ...")
    embedding_service = EmbeddingService(settings.openai)
    vectors = {}
    for gid, resps in groups.items():
        texts = [r.response_text for r in resps]
        vectors[gid] = await embedding_service.embed_texts(texts)

    # ── Cluster ───────────────────────────────────────────────────────────────
    print("Clustering ...")
    cluster_results: dict[str, QuestionClusterResult] = {}
    for gid, resps in groups.items():
        emb = vectors[gid]
        n = len(resps)
        min_cs = max(2, min(settings.clustering.min_cluster_size, n // 3))
        min_s = max(1, min(settings.clustering.min_samples, min_cs - 1))
        svc = ClusteringService(
            ClusteringSettings(
                min_cluster_size=min_cs,
                min_samples=min_s,
                metric=settings.clustering.metric,
                cluster_selection_method=settings.clustering.cluster_selection_method,
            )
        )
        cluster_results[gid] = svc.cluster_question(gid, resps, emb)

    # ── Analyse ───────────────────────────────────────────────────────────────
    print(f"\nAnalysing {len(cluster_results)} question groups with {settings.openai.analysis_model} ...\n")
    analysis_service = AnalysisService(settings.openai)

    for gid, result in cluster_results.items():
        question_text = groups[gid][0].question_text if groups[gid] else gid

        # When all responses are noise, build a synthetic cluster from all texts
        # so the LLM still gets something meaningful to analyze.
        if not result.clusters and result.noise_count > 0:
            all_quotes = [r.response_text for r in groups[gid] if r.response_text]
            synthetic_noise = ClusterAnalysis(
                cluster_id=-1,
                representative_quotes=all_quotes[:10],
                response_count=result.noise_count,
                percentage_of_total=100.0,
            )
            synthetic_result = result.model_copy(
                update={"clusters": [synthetic_noise], "noise_count": 0}
            )
        else:
            synthetic_result = result

        analyzed = await analysis_service.analyze_all_clusters(
            synthetic_result,
            question_text,
            include_noise=False,  # noise already wrapped above
        )

        print("=" * 65)
        print(f"[{gid}] {question_text[:70]}")
        print(f"  Clusters: {len(analyzed.clusters)}  |  Total responses: {analyzed.total_responses}")
        for c in analyzed.clusters:
            label = "NOISE GROUP" if c.cluster_id == -1 else f"Cluster {c.cluster_id}"
            print(f"\n  {label} ({c.response_count} responses, {c.percentage_of_total:.0f}%)")
            print(f"  Theme   : {c.theme}")
            print(f"  Summary : {c.summary}")
            print(f"  Insights:")
            for ins in c.insights:
                print(f"    - {ins}")

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"  Model used : {settings.openai.analysis_model}")
    print(f"  Groups     : {len(cluster_results)}")


asyncio.run(main())
