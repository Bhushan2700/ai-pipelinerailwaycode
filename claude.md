# Claude Instructions – AI Survey Report Pipeline

## 🎯 Objective

Build a scalable AI pipeline that:

1. Fetches survey data from FileMaker
2. Processes open-ended responses using AI (vectorization, clustering, analysis)
3. Combines with aggregated data
4. Generates a structured JSON report
5. Saves the final report back to FileMaker

---

## 🧠 System Architecture

* Backend Storage: FileMaker (Data API)
* API Layer: FastAPI
* Queue System: Redis
* Worker System: RQ (Redis Queue)
* AI Processing:

  * Embeddings: OpenAI
  * Clustering: HDBSCAN (local)
  * Analysis: OpenAI LLM

---

## ⚠️ Core Rules (VERY IMPORTANT)

1. FileMaker is ONLY for storage:

   * Fetch raw data
   * Store final results
   * DO NOT use FileMaker for processing

2. Always batch OpenAI requests:

   * Embeddings: batch size 50–100
   * Analysis: batch or per cluster

3. Clustering must be global per question:

   * NEVER cluster per batch

4. Each open question must be processed independently

5. Use queue-based processing:

   * No long-running synchronous APIs

---

## 📁 Project Structure

```
project/
├── app/
│   └── main.py
├── queue/
│   └── queue.py
├── workers/
│   └── process_question.py
├── tasks/
│   └── pipeline.py
├── services/
│   ├── filemaker_service.py
│   ├── embedding_service.py
│   ├── clustering_service.py
│   └── analysis_service.py
├── utils/
│   └── batching.py
├── config/
│   └── settings.py
```

---

## 🔄 Pipeline Flow

1. API endpoint triggered (`/generate-report`)
2. Job pushed to Redis queue
3. Worker executes pipeline:

   * Fetch data from FileMaker
   * Separate open questions
   * For each question:

     * Batch vectorization (OpenAI)
     * Merge vectors
     * Cluster (HDBSCAN)
     * Analyze clusters (OpenAI)
   * Combine results
   * Generate report JSON
   * Save report to FileMaker

---

## 🧩 Functional Requirements

### 1. FileMaker Service

* Authenticate using Data API
* Fetch records
* Save report JSON

### 2. Embedding Service

* Input: list of texts
* Output: list of vectors
* Must support batching

### 3. Clustering Service

* Input: vectors
* Output: cluster labels
* Use HDBSCAN
* Runs locally (no API)

### 4. Analysis Service

* Input: grouped responses
* Output: structured insights
* Use OpenAI LLM

### 5. Worker Logic

* Process one question at a time
* Use batching internally
* Return structured insights

### 6. Pipeline Task

* Orchestrates full flow
* Calls worker per question
* Combines results

---

## ⚡ Performance Requirements

* Must support:

  * 300 → 10,000 responses per question
* Use batching + parallel workers
* Avoid repeated FileMaker calls
* Minimize memory duplication

---

## 🔐 Environment Variables

```
OPENAI_API_KEY=
FILEMAKER_URL=
FILEMAKER_USERNAME=
FILEMAKER_PASSWORD=
REDIS_URL=
```

---

## 🧪 Development Rules

* Write clean, modular Python code
* Use async-safe patterns where needed
* Add logging for each step
* Handle API failures with retries
* Avoid hardcoding values

---

## 🚫 Do NOT Do

* Do NOT store embeddings in FileMaker
* Do NOT call OpenAI per single response
* Do NOT cluster per batch
* Do NOT block API with long processing

---

## ✅ Expected Output

Final report format:

```json
{
  "open_question_insights": [...],
  "aggregated_data": [...],
  "metadata": {...}
}
```

---

## 🧠 Claude Behavior Instructions

When generating code:

* Always follow modular structure
* Prefer clarity over cleverness
* Include comments explaining logic
* Keep functions small and testable
* Assume production-scale data

---

## 🚀 Future Extensions (Keep in Mind)

* Parallel processing per question
* Retry mechanisms for failed jobs
* Monitoring (logs, metrics)
* Migration to Celery/Temporal if needed

---
