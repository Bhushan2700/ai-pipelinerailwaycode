import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from core.config import get_settings
from services.filemaker import FileMakerClient

EVENT_ID = "951F9774-45EF-FD41-899A-4C57745021B5"


async def main():
    settings = get_settings()
    print(f"Connecting to FileMaker at {settings.filemaker.host} ...")

    async with FileMakerClient(settings.filemaker) as fm:
        print("Successfully logged in FileMaker")
        print(f"Fetching from N8N_SURVEY_EVENTS where ID_Event = {EVENT_ID} ...\n")

        data = await fm.fetch_event_responses(EVENT_ID)

        if not data:
            print("No records found.")
            return

        # ── Event metadata
        print("=" * 60)
        print("EVENT METADATA")
        print("=" * 60)
        print(json.dumps(data["meta"], indent=2))

        # ── Numeric_JSON
        numeric_questions = data["numeric_questions"]
        print("\n" + "=" * 60)
        print(f"NUMERIC_JSON — {len(numeric_questions)} structured questions")
        print("=" * 60)
        for qid, q in numeric_questions.items():
            explanations = [e for e in q.get("explanation", []) if e.strip()]
            print(f"\n  [{qid}] {q.get('question_text')} ({q.get('question_type')})")
            if "options" in q:
                print(f"       Aggregated options : {q['options']}")
            if "distribution" in q:
                print(f"       Distribution       : {q['distribution']}")
            print(f"       Open explanations  : {len(explanations)} non-empty")

        # ── Text_JSON
        text_questions = data["text_questions"]
        print("\n" + "=" * 60)
        print(f"TEXT_JSON — {len(text_questions)} open-ended questions")
        print("=" * 60)
        for q in text_questions:
            responses = q.get("responses", [])
            print(f"\n  [{q['question_id']}] {q['question_text']}")
            print(f"       Total responses: {len(responses)}")
            for r in responses:
                print(f"         - {r['answer'][:120]}")


asyncio.run(main())
