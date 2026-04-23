"""
Step-by-step test: Fetch -> Vectorize -> Cluster -> Data Analysis
Prints every response, vector shapes, cluster assignments, and deep analysis.
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from core.config import get_settings, ClusteringSettings
from services.analysis import AnalysisService
from models.survey import SurveyResponse, QuestionType
from services.filemaker import FileMakerClient
from services.embedding import EmbeddingService
from services.clustering import ClusteringService

EVENT_ID = "951F9774-45EF-FD41-899A-4C57745021B5"

SEP  = "=" * 65
SEP2 = "-" * 65


def build_text_responses(q: dict) -> tuple[str, str, list[SurveyResponse]]:
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
    return qid, qtext, resps


def build_explanation_responses(qid: str, q: dict) -> tuple[str, str, list[SurveyResponse]]:
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
    return gid, qtext, resps


async def main():
    settings = get_settings()

    # ══════════════════════════════════════════════════════════════
    # STEP 1 — FETCH FROM FILEMAKER
    # ══════════════════════════════════════════════════════════════
    print(SEP)
    print("STEP 1: FETCH FROM FILEMAKER")
    print(SEP)

    async with FileMakerClient(settings.filemaker) as fm:
        data = await fm.fetch_event_responses(EVENT_ID)

    if not data:
        print("No data returned.")
        return

    print(f"  Survey     : {data.get('survey_name', '—')}")
    print(f"  Industry   : {data.get('industry', '—')}")
    print(f"  Record ID  : {data.get('record_id')}")

    text_qs    = data.get("text_questions", [])
    numeric_qs = data.get("numeric_questions", {})
    print(f"  Open-ended questions : {len(text_qs)}")
    print(f"  Numeric questions    : {len(numeric_qs)}")

    # Print every open-ended response
    print(f"\n{SEP2}")
    print("  OPEN-ENDED RESPONSES")
    print(SEP2)
    for q in text_qs:
        print(f"\n  [{q['question_id']}] {q['question_text']}")
        for i, r in enumerate(q.get("responses", []), 1):
            ans = r.get("answer", "").strip()
            if ans:
                print(f"    {i:02d}. {ans}")

    # Print every explanation text
    print(f"\n{SEP2}")
    print("  EXPLANATION TEXTS (from numeric questions)")
    print(SEP2)
    for qid, q in numeric_qs.items():
        exps = [e.strip() for e in q.get("explanation", []) if e.strip()]
        if exps:
            print(f"\n  [{qid}] {q.get('question_text', '')}")
            for i, e in enumerate(exps, 1):
                print(f"    {i:02d}. {e}")

    # ── Build response groups ─────────────────────────────────────
    groups: dict[str, tuple[str, list[SurveyResponse]]] = {}
    for q in text_qs:
        qid, qtext, resps = build_text_responses(q)
        if resps:
            groups[qid] = (qtext, resps)
    for qid, q in numeric_qs.items():
        gid, qtext, resps = build_explanation_responses(qid, q)
        if resps:
            groups[gid] = (qtext, resps)

    print(f"\n  Total groups to process : {len(groups)}")
    for gid, (_, resps) in groups.items():
        print(f"    {gid:12s}  {len(resps)} responses")

    # ══════════════════════════════════════════════════════════════
    # STEP 2 — VECTORIZE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 2: VECTORIZE (OpenAI text-embedding-3-small)")
    print(SEP)

    embedding_service = EmbeddingService(settings.openai)
    vectors: dict = {}

    for gid, (qtext, resps) in groups.items():
        texts = [r.response_text for r in resps]
        print(f"\n  [{gid}]")
        print(f"  Question : {qtext[:70]}")
        print(f"  Sending {len(texts)} texts to OpenAI ...")
        emb = await embedding_service.embed_texts(texts)
        vectors[gid] = emb
        print(f"  Result   : shape {emb.shape}  dtype={emb.dtype}")
        print(f"  Sample vector[0][:6] = {emb[0][:6].tolist()}")

    total_vectors = sum(v.shape[0] for v in vectors.values())
    print(f"\n  Total vectors produced : {total_vectors}")

    # ══════════════════════════════════════════════════════════════
    # STEP 3 — CLUSTER
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 3: CLUSTER (HDBSCAN)")
    print(SEP)

    for gid, (qtext, resps) in groups.items():
        emb = vectors[gid]
        n   = len(resps)

        min_cs = max(2, min(settings.clustering.min_cluster_size, n // 3))
        min_s  = max(1, min(settings.clustering.min_samples, min_cs - 1))

        svc    = ClusteringService(ClusteringSettings(
            min_cluster_size=min_cs,
            min_samples=min_s,
            metric=settings.clustering.metric,
            cluster_selection_method=settings.clustering.cluster_selection_method,
        ))
        result = svc.cluster_question(gid, resps, emb)

        print(f"\n  [{gid}] {qtext[:70]}")
        print(f"  Responses: {result.total_responses}  |  Clusters: {len(result.clusters)}  |  Noise: {result.noise_count}")

        # Print each response with its cluster label
        for lbl, resp in zip(result.cluster_labels, resps):
            tag = f"Cluster {lbl.cluster_id}" if lbl.cluster_id != -1 else "NOISE"
            print(f"    [{tag:10s}] (p={lbl.membership_probability:.2f})  {resp.response_text[:80]}")

        if result.clusters:
            print(f"  Cluster summary:")
            for c in result.clusters:
                print(f"    Cluster {c.cluster_id}: {c.response_count} responses ({c.percentage_of_total:.1f}%)")
                for q in c.representative_quotes:
                    print(f"      -> {q[:80]}")

    numeric_questions = data.get("numeric_questions", {})

    # ══════════════════════════════════════════════════════════════
    # STEP 4a — PRINT NUMERIC_JSON (aggregated response values)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 4a: NUMERIC_JSON — AGGREGATED RESPONSE DATA")
    print(SEP)
    for qid, q in numeric_questions.items():
        print(f"\n  [{qid}] {q.get('question_text', '')}")
        print(f"  Type    : {q.get('question_type', '—')}  |  Total: {q.get('total_answers', '—')}")
        options = q.get("options", {})
        total_ans = q.get("total_answers", 1) or 1
        if options:
            print(f"  Options :")
            for opt, count in sorted(options.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
                if isinstance(count, (int, float)):
                    pct = round(count / total_ans * 100, 1)
                    print(f"    {opt:45s}  {count:3d}  ({pct:.1f}%)")
                else:
                    print(f"    {opt:45s}  {json.dumps(count)}")
        else:
            print(f"  Options : (no aggregated option data)")

    # ══════════════════════════════════════════════════════════════
    # STEP 4b — MERGE numeric data + explanation clusters
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 4b: MERGE — Numeric aggregation + Explanation texts")
    print(SEP)

    # questions_for_analysis: { qid: (question_text, [response_texts], numeric_ctx | None) }
    questions_for_analysis: dict[str, tuple[str, list[str], dict | None]] = {}

    # Open-ended questions — no numeric context
    open_qids = {q["question_id"] for q in data.get("text_questions", [])}
    for gid, (qtext, resps) in groups.items():
        base_qid = gid.replace("_exp", "")
        if gid in open_qids:
            questions_for_analysis[gid] = (
                qtext,
                [r.response_text for r in resps if r.response_text],
                None,
            )

    # Numeric questions — merge options + explanations
    for qid, q in numeric_questions.items():
        exp_texts = [e.strip() for e in q.get("explanation", []) if e.strip()]
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

    # Print the merged view
    for qid, (qtext, resps, num_ctx) in questions_for_analysis.items():
        print(f"\n  [{qid}] {qtext[:70]}")
        if num_ctx:
            print(f"  Type    : {num_ctx['question_type']}  |  Total: {num_ctx['total_answers']}")
            total_ans = num_ctx["total_answers"] or 1
            opts = num_ctx["options"]
            if opts:
                print(f"  Options :")
                for opt, count in sorted(opts.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
                    if isinstance(count, (int, float)):
                        pct = round(count / total_ans * 100, 1)
                        print(f"    {opt:45s}  {count:3d}  ({pct:.1f}%)")
                    else:
                        print(f"    {opt:45s}  {json.dumps(count)}")
            else:
                print(f"  Options : (no aggregated data)")
            print(f"  Explanations ({len(resps)}):")
            for i, e in enumerate(resps, 1):
                print(f"    {i}. {e[:100]}")
        else:
            print(f"  Type    : open-ended  |  Responses: {len(resps)}")
            for i, r in enumerate(resps, 1):
                print(f"    {i}. {r[:100]}")

    # ══════════════════════════════════════════════════════════════
    # STEP 4c — DATA ANALYST AGENT (single combined call)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 4c: DATA ANALYST AGENT (gpt-4o-mini) — ONE call for ALL questions")
    print(SEP)

    analysis_service = AnalysisService(settings.openai)
    print(f"\n  Sending all {len(questions_for_analysis)} questions in one GPT call ...\n")
    analyses = await analysis_service.analyze_all_questions(questions_for_analysis)

    for result in analyses:
        qid = result.get("question_id", "")
        s   = result.get("summary", {})

        print(SEP2)
        print(f"  [{qid}]  {s.get('question_text','')[:70]}")
        print(f"  Type      : {s.get('question_type', '—')}")
        print(f"  Total     : {s.get('total_responses', '—')}")
        print(f"  Sentiment : {s.get('sentiment', '—').upper()}")
        print(f"\n  summary.overview:")
        print(f"    {s.get('overview', s.get('summary','—'))}")
        print(f"\n  summary.majority_view:")
        print(f"    {s.get('majority_view','—')}")
        print(f"\n  summary.minority_view:")
        print(f"    {s.get('minority_view','—')}")
        print(f"\n  summary.key_findings:")
        for f in s.get("key_findings", []):
            print(f"    - {f}")
        print(f"\n  summary.response_patterns:")
        for p in s.get("response_patterns", []):
            print(f"    * {p}")
        print()

    print(SEP)
    print(f"  Questions analysed : {len(analyses)}")
    print("DONE — Fetch -> Vectorize -> Cluster -> Analysis complete")
    print(SEP)

    # ══════════════════════════════════════════════════════════════
    # STEP 5 — REPORT GENERATION + RISK AGENT (parallel) → MERGE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 5: REPORT GENERATION AGENT + RISK AGENT (gpt-4o-mini, parallel)")
    print(SEP)

    from services.analysis import merge_report_outputs

    survey_meta = {
        "survey_id":          data.get("survey_id", ""),
        "survey_name":        data.get("survey_name", ""),
        "industry":           data.get("industry", ""),
        "record_id":          data.get("record_id", ""),
        "survey_description": data.get("survey_description", ""),
        "survey_recipients":  data.get("survey_recipients", "0"),
    }

    print("\n  Running Report Writer and Risk agents in parallel ...\n")

    report_out, risk_out = await asyncio.gather(
        analysis_service.generate_report(analyses, survey_meta),
        analysis_service.generate_risk_and_action_plan(analyses),
    )

    # ── Print Report Writer output ────────────────────────────────
    print(SEP2)
    print("  REPORT WRITER AGENT OUTPUT")
    print(SEP2)
    print(json.dumps(report_out, indent=2, ensure_ascii=False))

    # ── Print Risk Agent output ───────────────────────────────────
    print(f"\n{SEP2}")
    print("  RISK AGENT OUTPUT")
    print(SEP2)
    print(json.dumps(risk_out, indent=2, ensure_ascii=False))

    # ── Merge (same logic as JS pipeline) ────────────────────────
    print(f"\n{SEP}")
    print("  MERGING OUTPUTS  { ...reportJSON, ...riskJSON }")
    print(SEP)

    merged, record_id = merge_report_outputs(report_out, risk_out)

    print(json.dumps(merged, indent=2, ensure_ascii=False))

    print(f"\n{SEP}")
    print("FINAL OUTPUT")
    print(SEP)
    print(f"  record_id  : {record_id}")
    print(f"  Top-level keys in merged JSON:")
    for k in merged.keys():
        print(f"    - {k}")
    print(f"\n  Total keys : {len(merged)}")
    print(f"\n  --> Storing to FileMaker record {record_id} ...")
    print(SEP)

    # ══════════════════════════════════════════════════════════════
    # STEP 6 — STORE FINAL REPORT TO FILEMAKER
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 6: STORE FINAL REPORT TO FILEMAKER")
    print(SEP)

    fm_record_id = data.get("record_id", "")
    if not fm_record_id:
        print("  ERROR: no record_id in fetched data — skipping FileMaker write")
    else:
        report_payload = json.dumps(merged, ensure_ascii=False)
        print(f"  Layout    : N8N_SURVEY_EVENTS")
        print(f"  Record ID : {fm_record_id}  (received from FileMaker at Step 1)")
        print(f"  Field     : asJSON_Report")
        print(f"  Payload   : {len(report_payload):,} characters\n")

        async with FileMakerClient(settings.filemaker) as fm:
            await fm.update_record(
                layout="N8N_SURVEY_EVENTS",
                record_id=fm_record_id,
                field_data={"asJSON_Report": report_payload},
            )

        print(f"  SUCCESS — report written to N8N_SURVEY_EVENTS record {fm_record_id}")

    print(SEP)
    print("PIPELINE COMPLETE")
    print("Fetch -> Vectorize -> Cluster -> Analyse -> Report + Risk -> Merge -> Store")
    print(SEP)


asyncio.run(main())
