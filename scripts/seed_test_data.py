"""
Generates a realistic FileMaker-shaped survey fixture for testing.
Usage: python scripts/seed_test_data.py
Output: tests/fixtures/survey_data.json
"""
import json
import os
import random

OPEN_ENDED_TEXTS = [
    "The service was excellent and staff were very helpful",
    "Waiting times were too long and frustrating",
    "Clean and well maintained facility overall",
    "Staff communication could be improved significantly",
    "Very satisfied with the quality of care provided",
    "The process was confusing and hard to navigate",
    "Friendly and professional team, great experience",
    "Long queues at reception ruined my visit",
    "Everything was handled quickly and efficiently",
    "I felt well informed throughout the whole process",
]


def generate_records(n: int = 300, questions: int = 2) -> list[dict]:
    records = []
    record_id = 1
    for q_idx in range(1, questions + 1):
        for _ in range(n):
            is_open = q_idx == 1
            record = {
                "recordId": str(record_id),
                "fieldData": {
                    "respondent_id": f"R{record_id:04d}",
                    "question_id": f"Q{q_idx}",
                    "question_text": "How was your overall experience?" if is_open else "Rate your satisfaction (1-5)",
                    "question_type": "open_ended" if is_open else "likert",
                    "response_text": random.choice(OPEN_ENDED_TEXTS) if is_open else None,
                    "response_value": None if is_open else float(random.randint(1, 5)),
                },
            }
            records.append(record)
            record_id += 1
    return records


if __name__ == "__main__":
    os.makedirs("tests/fixtures", exist_ok=True)
    records = generate_records()
    with open("tests/fixtures/survey_data.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Generated {len(records)} records → tests/fixtures/survey_data.json")
