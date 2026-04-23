"""
Step 2: Vectorize open text responses fetched from FileMaker.

What gets vectorized:
  - Text_JSON  -> pure open-ended answers (Q6, Q10) — every response answer
  - Numeric_JSON -> non-empty explanation texts per question (Q1–Q9)
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from core.config import get_settings
from services.filemaker import FileMakerClient
from services.embedding import EmbeddingService

EVENT_ID = "951F9774-45EF-FD41-899A-4C57745021B5"


def extract_texts(data: dict) -> dict[str, list[str]]:
    """
    Extracts texts to vectorize grouped by question_id.

    Returns:
        { "Q6": ["answer1", "answer2", ...], "Q10": [...], "Q1_exp": [...], ... }
    """
    texts: dict[str, list[str]] = {}

    # ── Text_JSON: pure open-ended — vectorize every answer
    for q in data.get("text_questions", []):
        qid = q["question_id"]
        answers = [r["answer"].strip() for r in q.get("responses", []) if r.get("answer", "").strip()]
        if answers:
            texts[qid] = answers

    # ── Numeric_JSON: vectorize non-empty explanation texts
    for qid, q in data.get("numeric_questions", {}).items():
        explanations = [e.strip() for e in q.get("explanation", []) if e.strip()]
        if explanations:
            texts[f"{qid}_exp"] = explanations

    return texts


async def main():
    settings = get_settings()
    print("Connecting to FileMaker ...")

    async with FileMakerClient(settings.filemaker) as fm:
        print("Successfully logged in FileMaker")
        data = await fm.fetch_event_responses(EVENT_ID)

    if not data:
        print("No data fetched.")
        return

    # Extract texts grouped by question
    texts_by_question = extract_texts(data)

    print(f"\nTexts to vectorize across {len(texts_by_question)} groups:")
    for qid, texts in texts_by_question.items():
        print(f"  {qid:12s} -> {len(texts)} texts")

    total = sum(len(t) for t in texts_by_question.values())
    print(f"\nTotal texts to embed: {total}")

    # Vectorize each group
    embedding_service = EmbeddingService(settings.openai)
    vectors: dict[str, list] = {}

    print("\nVectorizing ...")
    for qid, texts in texts_by_question.items():
        print(f"  [{qid}] embedding {len(texts)} texts ...", end=" ", flush=True)
        embeddings = await embedding_service.embed_texts(texts)
        vectors[qid] = embeddings
        print(f"done -> shape {embeddings.shape}")

    print("\n" + "=" * 50)
    print("VECTORIZATION COMPLETE")
    print("=" * 50)
    for qid, emb in vectors.items():
        print(f"  {qid:12s} -> {emb.shape[0]} vectors x {emb.shape[1]} dims")

    print(f"\nTotal vectors produced: {sum(e.shape[0] for e in vectors.values())}")


asyncio.run(main())
