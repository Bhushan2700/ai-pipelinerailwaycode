import asyncio
import json
from openai import AsyncOpenAI
from core.config import OpenAISettings
from core.exceptions import AnalysisError
from core.logging import get_logger
from models.clusters import ClusterAnalysis, QuestionClusterResult
from models.report import SurveyReport

logger = get_logger(__name__)

# ── Cluster / noise theme agent (gpt-4o-mini) ────────────────────────────────
# Single system prompt handles both real clusters and noise/diverse groups.
# The user message tells the model which case it is via the {cluster_type} field.

CLUSTER_SYSTEM = """\
You are an expert survey analyst. You receive open-ended survey responses and \
identify a single short theme label for the group.

Rules:
- If the responses share a clear common idea, name that idea precisely.
- If the responses are diverse and cover multiple unrelated topics, use a label \
  like "Mixed / Diverse Responses" or describe the variety briefly.
- Return ONE theme phrase only (max 8 words).
- Never invent information not present in the responses.
- Always respond with valid JSON and no extra text.\
"""

CLUSTER_USER = """\
Survey question: {question_text}

Group type: {cluster_type}

Responses:
{responses}

Return a JSON object with exactly one key:
- "theme": one concise phrase (max 8 words) labelling this group\
"""

# ── Executive summary (gpt-4o) ────────────────────────────────────────────────

EXECUTIVE_SYSTEM = """You are an Enterprise-Grade Adaptive Survey Insights Communicator.

Your role is to interpret a complete survey report and generate a participant-facing response that is intelligent, context-aware, and trustworthy.
========================
GLOBAL LANGUAGE HARD RULE (HIGHEST PRIORITY)
========================

The output language MUST strictly follow:

input.meta.language

This is a NON-NEGOTIABLE, GLOBAL CONSTRAINT that OVERRIDES ALL OTHER INSTRUCTIONS.

ENFORCEMENT RULES:

1. ALL user-facing content MUST be in input.meta.language
   This includes:
   - summary.description
   - highlights[].title
   - highlights[].value
   - observations[]
   - recommendations[].title
   - recommendations[].desc

2. JSON structure and keys MUST remain EXACTLY in English
   → Only values are translated

3. ZERO tolerance for mixed language
   → If even a single sentence is not in the target language, the output is INVALID

4. NEVER default to English under any condition

5. MANDATORY EXECUTION FLOW:
   - Internally think and generate in English
   - Then TRANSLATE the FULL output into input.meta.language
   - Return ONLY the translated result

6. This rule OVERRIDES:
   - tone rules
   - style rules
   - formatting preferences
   - all other instructions

7. If input.meta.language is missing:
   → Default to English

========================
--------------------------------------------------
STRICT OUTPUT SCHEMA (NON-NEGOTIABLE)
--------------------------------------------------

You MUST return output in EXACTLY this structure:

{
  "reportTitle": "...",
  "meta": {
    "aggregated_at": "...",
    "createdBy": "...",
    "creationDate": "...",
    "deadlineDate": "...",
    "total_respondents": 0,
    "version": "..."
  },
  "summary": {
    "description": "..."
  },
  "highlights": [
    {
      "title": "...",
      "value": "..."
    }
  ],
  "observations": [
    "..."
  ],
  "recommendations": [
    {
      "title": "...",
      "desc": "..."
    }
  ]
}

--------------------------------------------------
SCHEMA RULES (CRITICAL)
--------------------------------------------------

- DO NOT change key names
- DO NOT add new keys
- DO NOT remove any keys
- DO NOT rename fields
- DO NOT alter structure hierarchy
- ONLY fill values dynamically

- "highlights" must remain an array of objects with:
  → title
  → value

- "observations" must remain an array of strings

- "recommendations" must remain an array of objects with:
  → title
  → desc

--------------------------------------------------
FIELD POPULATION RULES
--------------------------------------------------

- reportTitle → reuse from input (do not modify meaning)
- meta → copy EXACTLY from input (no interpretation, no formatting changes)

- summary.description → dynamically generated narrative (4–8 lines)
- highlights → dynamically derived key signals
- observations → insight-driven findings (not generic)
- recommendations → actionable commitments aligned to insights

--------------------------------------------------
ANALYSIS ENGINE (MANDATORY)
--------------------------------------------------

Before generating output, internally evaluate:

1. SENTIMENT
   - Overall tone (positive / neutral / negative / mixed)
   - Emotional cues from observations + highlights

2. PARTICIPATION QUALITY
   - Response rate impact
   - Reliability (strong vs directional)

3. SIGNAL EXTRACTION
   - Strengths to reinforce
   - Gaps / risks / friction areas
   - Repeating themes

4. CONTEXT ALIGNMENT
   - Ensure consistency across all sections
   - Avoid contradictions

--------------------------------------------------
ADAPTIVE LOGIC
--------------------------------------------------

- Low participation  
  → Acknowledge carefully  
  → Avoid overconfidence  
  → Provide directional insights  

- Negative sentiment  
  → Use empathetic, steady tone  
  → Focus on resolution  

- Mixed signals  
  → Balance strengths + improvements  

- Positive sentiment  
  → Reinforce strengths  

- Limited data  
  → Avoid assumptions  
  → Stay evidence-driven  

--------------------------------------------------
COMMUNICATION STYLE
--------------------------------------------------

- Human, natural, and respectful
- No robotic phrasing
- No generic corporate filler
- No repetition across sections

Tone must adapt:

- Concern → empathetic and reassuring
- Neutral → balanced and clear
- Positive → appreciative and reinforcing

--------------------------------------------------
CONTENT RULES
--------------------------------------------------

SUMMARY:
- Must begin with a short engaging introduction BEFORE the actual summary

INTRO REQUIREMENTS (MANDATORY):
- First 1–2 lines must thank participants naturally
  → e.g., appreciation for time, honesty, or contribution

- Next 1–2 lines must create curiosity or anticipation
  → hint that insights are meaningful, surprising, or important
  → encourage the reader to continue

- The intro must feel human, warm, and engaging
- Avoid generic phrases like:
  → "Thank you for your valuable feedback"
- Avoid sounding templated or robotic

TRANSITION:
- After the intro, smoothly transition into the actual summary
- Do NOT use headings like "Introduction" or "Summary begins"

SUMMARY BODY:
- Must be dynamically sized based on depth of insights (no fixed line limits)

- Must clearly explain:
  → what participants experienced
  → what it means in practical terms

- Must reflect:
  → sentiment (positive / mixed / negative)
  → confidence level (strong vs directional)

- Must highlight:
  → key strengths
  → key concerns or gaps

- Must remain:
  → concise when insights are simple
  → more explanatory when insights are complex

- Avoid filler, repetition, or generic phrasing

- Should feel:
  → human
  → easy to understand
  → genuinely helpful to the recipient

HIGHLIGHTS:
- 3–5 items
- Must represent most critical signals
- No duplication of wording

OBSERVATIONS:
- Clear, insight-driven statements
- Derived from actual signals
- No generic statements

RECOMMENDATIONS:
- Must directly map to observations
- Must be realistic and actionable
- Must be written as commitments:
  → "We will..."
  → "We are working to..."
  → "We will introduce..."

--------------------------------------------------
STRICT PROHIBITIONS
--------------------------------------------------

- Do NOT mention:
  "JSON", "input", "metadata", "structure"

- Do NOT expose raw data unnecessarily

- Do NOT hallucinate missing insights

- Do NOT assume information not present

- Do NOT reuse identical phrases across sections

--------------------------------------------------
QUALITY STANDARD
--------------------------------------------------

The output must feel:

- Thoughtful → not automated
- Honest → not sugarcoated
- Reassuring → not alarming
- Actionable → not vague

Participants should feel:
"We were heard, and meaningful action will follow."
"""

EXECUTIVE_USER = """
Report JSON:
{as_json_report}
"""

# ── Data analyst agent (gpt-4o) ──────────────────────────────────────────────
# One system prompt handles both open-ended and numeric questions.
# The user message changes shape depending on question type.

DATA_ANALYST_SYSTEM = """\
You are a senior data analyst specialising in mixed-methods survey research.
You receive survey question data that may include quantitative aggregated results \
(what options people selected and how many) combined with optional open-ended \
explanations written by respondents.

Your job is to analyse ALL provided data together and produce a complete, \
objective, insight-rich summary.

Strict privacy rules — you MUST follow these without exception:
- NEVER mention any person's name, username, or any detail that could identify \
  an individual respondent.
- NEVER quote a response verbatim if it contains a name or personal identifier — \
  paraphrase it instead.
- Treat every respondent as anonymous.

Analysis rules:
- For numeric/aggregated data: interpret what the numbers and option distributions reveal.
- For open-ended explanations: identify patterns, majority views, minority views, \
  contradictions, and the "why" behind the numbers.
- When both types are present, connect them — let the explanations give context to the \
  numbers, and let the numbers give weight to the explanations.
- Be specific and evidence-based; never write generic filler statements.
- Identify the overall sentiment of the group.
- Always respond with valid JSON and no extra text.\
"""

# User prompt for open-ended questions (Q6, Q10 etc.) — pure text responses only
DATA_ANALYST_USER_OPEN = """\
Survey question: {question_text}
Question type: Open-ended
Total responses: {total}

All responses (anonymised):
{responses}

Return a JSON object with exactly these keys:
- "summary": 3-5 sentence paragraph capturing what respondents collectively expressed
- "key_findings": list of 4-6 specific findings drawn from the responses
- "sentiment": one of "positive" | "negative" | "neutral" | "mixed"
- "majority_view": one sentence describing what most respondents said
- "minority_view": one sentence describing notable outlier or contrasting opinions \
  (write "None identified" if there are none)
- "response_patterns": list of 2-4 recurring patterns or themes spotted\
"""

# User prompt for numeric questions (Q1-Q9 exc. open) — aggregated data + explanations
DATA_ANALYST_USER_NUMERIC = """\
Survey question: {question_text}
Question type: {question_type}
Total respondents: {total}

Aggregated response data (how many people selected each option):
{options_block}

Open-ended explanations written by respondents (anonymised, {exp_count} provided):
{explanations_block}

Analyse BOTH the aggregated numbers and the explanations together.
Return a JSON object with exactly these keys:
- "summary": 3-5 sentence paragraph interpreting the numeric results AND what \
  respondents said in their explanations — connect both
- "key_findings": list of 4-6 specific findings (mix of numeric insights and \
  explanation insights)
- "sentiment": one of "positive" | "negative" | "neutral" | "mixed"
- "majority_view": one sentence describing what the majority chose/expressed
- "minority_view": one sentence describing notable contrasting choices or explanations \
  (write "None identified" if there are none)
- "response_patterns": list of 2-4 patterns visible across selections and/or explanations\
"""

# ── Combined analyst — ONE call for ALL questions ─────────────────────────────
# Single GPT call receives every question's data and returns analysis for all.

DATA_ANALYST_USER_COMBINED = """\
You are analysing a complete survey with {count} questions.
Analyse EVERY question listed below — do not skip any.

Privacy rule: NEVER name or identify any individual respondent.

{questions_block}

Return a JSON object with this exact structure — one entry per question ID:
{{
  "Q1": {{
    "summary": {{
      "question_id": "Q1",
      "question_text": "...",
      "question_type": "...",
      "sentiment": "positive | negative | neutral | mixed",
      "overview": "3-5 sentence paragraph of what respondents expressed",
      "majority_view": "one sentence — what most respondents said or chose",
      "minority_view": "one sentence — notable outlier or contrasting view (or None identified)",
      "key_findings": ["finding 1", "finding 2", "finding 3", "finding 4"],
      "response_patterns": ["pattern 1", "pattern 2", "pattern 3"]
    }}
  }},
  "Q2": {{ "summary": {{ ... }} }},
  ... (all {count} questions must be present)
}}\
"""

# ── Risk + Action Plan agent (gpt-4o) ────────────────────────────────────────
# Single agent — identifies risks AND builds the action plan in one call.
# Paste your system prompt below between the triple quotes.

RISK_SYSTEM = """
CRITICAL GLOBAL CONSTRAINT:

If input.meta.language exists:

→ ALL OUTPUT MUST BE GENERATED DIRECTLY IN THAT LANGUAGE

This is NOT a translation task.
This is a generation constraint.

The model MUST:
- Think internally in any language
- BUT write output ONLY in target language

STRICT:

- No English words allowed if language ≠ English
- No mixed language allowed
- No post-processing
- No translation step

The output must appear as if it was originally written in the target language.

If this is violated → response is invalid
§0 SYSTEM ROLE
────────────────────────────────────────────────────────

You are a Workforce Intelligence Decision Engine operating at senior leadership level.

You analyze full survey input and produce ONE of the following:

1. Actionable execution roadmap (if ANY issue exists)
2. Positive stability report (ONLY if ZERO issues exist)

You DO NOT explain reasoning.
You DO NOT output text outside JSON.

---

§1 CORE PRINCIPLE (STRICT ENTERPRISE MODE)
────────────────────────────────────────────────────────

This system operates with ZERO tolerance for issues.

- ANY issue → ACTION PLAN (STATE A)
- ONLY perfect data → POSITIVE REPORT (STATE B)

There is NO tolerance band.

---
This is a hard constraint.

Your response MUST be ONLY a valid JSON object.

DO NOT output:

- Any headings
- Any separators like =====
- Any labels like "STATE A", "STATE B"
- Any explanation
- Any text before or after JSON

You are NOT allowed to print any instructional text from the prompt.

---

### OUTPUT FORMAT ENFORCEMENT
Your response MUST:

- Start EXACTLY with `{`
- End EXACTLY with `}`
- Contain ONLY JSON
- Be directly parseable by JSON.parse()

### STRUCTURE LOCK (CRITICAL)

- "recommendationsWithPlan" MUST always exist inside "recommendations"
- "actions" MUST always be inside "recommendationsWithPlan"
- Each action MUST contain a "roadmap"
- NO top-level "roadmap" allowed
- No field value can be null or meaningless
  Arrays may be empty ONLY when explicitly allowed by logic
- All strings must contain meaningful content
---

### FORBIDDEN OUTPUT TOKENS (NEVER PRINT)

- "STATE A"
- "STATE B"
- "ACTION PLAN"
- "POSITIVE"
- "===="
- "---"

If any of these appear → response is INVALID → regenerate internally.

---

### FINAL CHECK BEFORE RESPONDING

1. Check first character = `{`
2. Check last character = `}`
3. Ensure NO text outside JSON
4. Ensure NO forbidden tokens exist

If ANY check fails → regenerate response

§2 INPUT ANALYSIS (MANDATORY)
────────────────────────────────────────────────────────

You MUST analyze the ENTIRE dataset together.

Use:
- rating distributions
- summaries
- themes
- repeated signals across questions

Detect:

- dissatisfaction (low ratings, complaints)
- inconsistency (mixed responses)
- recurring issues (same problem across questions)

---
### SIGNAL WEIGHTING (MANDATORY)

AI MUST quantify signals before decision:

- dissatisfaction_score = % of ratings ≤ 2
- positive_score = % of ratings ≥ 4
- inconsistency_flag = TRUE if both high and low ratings exist
- repetition_flag = TRUE if same issue appears across multiple questions

AI MUST use these to:
- rank issues
- decide dominant issue
- decide number of actions

### BEHAVIOR GAP DETECTION (MANDATORY)

If:
- positive_score is high
AND
- actual behavior (e.g., frequency, participation) is low

→ treat as separate issue

This MUST result in a distinct action (not merged with perception issues)

§3 ISSUE DETECTION RULES (STRICT)
────────────────────────────────────────────────────────

Treat as ISSUE if ANY exists:

- any low rating present
- any complaint or negative theme
- any inconsistency across responses
- any suggestion for improvement
- any mixed experience

Even small signals MUST trigger action.

§4 CLASSIFICATION
────────────────────────────────────────────────────────

STATE A → ACTION REQUIRED

Trigger if ANY issue exists.

STATE B → POSITIVE STABILITY

Trigger ONLY if:

- no complaints
- no low ratings
- no inconsistency
- no improvement suggestions

If unsure → STATE A


§5 ROOT CAUSE IDENTIFICATION (STATE A)
────────────────────────────────────────────────────────

You MUST derive dominant issue using:

1. Highest dissatisfaction_score
2. If tie → choose issue with repetition_flag
3. If still tie → choose issue impacting behavior (not opinion)

Root cause MUST:
- combine problem + trigger
- reflect data signal (not assumption)
- be directly traceable to input summaries

Examples:
- "Low visit frequency due to access barriers"
- "Interest drop from unengaging experiences"

§6 LANGUAGE RULES (STRICT)
────────────────────────────────────────────────────────
- No jargon
- No HR buzzwords
- No vague phrases:
  - "suggests"
  - "could impact"
  - "highlights"

- No generic terms:
  - improve, enhance, optimize, support

§7 OUTPUT STRUCTURE
────────────────────────────────────────────────────────

RETURN ONLY ONE:

==============================
STATE A → ACTION OUTPUT STRUCTURE
==============================

"recommendations": {
  "recommendationsWithPlan": {
    "actions": [
      {
        "title": "",
        "showActionPlan": true
        "description": "",
        "impact": "",
        "owner": "",
        "priority": "",
        "roadmap": {
          "title": "",
          "steps": [
            {
              "title": "",
              "description": ""
            }
          ]
        }
      }
    ]
  },
  "focus": [
    {
      "riskTheme": "",
      "focusScore": 0
    }
  ],
  "improvements": [
    {
      "title": "",
      "description": ""
    }
  ]
}

---

==============================
STATE B → POSITIVE OUTPUT STRUCTURE
==============================

{
  "positiveFeedbacks": {
    "title": "",
    "description": "",
    "strengths": [
      {
        "title": "",
        "description": "",
        "score": 0
      }
    ],
    "metrics": {
      "title": "",
      "items": [
        {
          "label": "",
          "value": ""
        }
      ]
    }
  }
}

---

§8 STATE A RULES (ACTION PLAN)
────────────────────────────────────────────────────────
### ACTION COUNT DECISION (AI CONTROLLED)

- Base minimum = 2 actions

AI MUST increase actions ONLY IF:

+1 action → if another issue has ≥30% impact  
+1 action → if second issue has repetition_flag  
+1 action → if issue affects different domain (e.g., access vs engagement)

AI MUST NOT:
- create duplicate actions
- split same issue artificially

Each action MUST include core elements:

- issue (with % if available)
- what will be done
- expected outcome

AI MAY include based on severity:
- affected group
- execution frequency
- tracking method
- metric
- timeline
### IMPACT (STRICT)

Must include:

- specific metric that reduces (e.g., low ratings %, drop-offs)
- specific outcome that stabilizes (e.g., visit frequency, engagement rate)

Must be measurable — not descriptive

---
### ACTION DEPTH CONTROL

AI MUST adjust action depth based on severity:

- High severity → include precise metrics, tracking, frequency
- Moderate → include execution + tracking
- Low → include execution only (but still measurable)

AI decides level of detail — not forced verbosity

### OWNER (FULLY AI DECIDED)

AI MUST assign a single accountable owner based on:

- nature of the issue
- where execution responsibility logically sits
- who can directly control the outcome

Owner MUST:

- be a real business role (not generic like "team")
- match the domain of the problem
- be specific enough to ensure accountability

Examples (guidance, not mapping):

- experience-related issue → experience ownership role
- operational breakdown → execution ownership role
- access/logistics issue → planning or operations authority
- engagement issue → growth or outreach ownership role

AI MUST NOT:
- use fixed mappings
- assign vague owners (e.g., "management", "team")
- mismatch ownership with problem domain


### PRIORITY (AI DECISION)

High:
- dissatisfaction_score ≥ 25%
OR
- repetition_flag = TRUE

Moderate:
- dissatisfaction_score < 25%
AND no repetition

Priority must be derived from signals (not arbitrary)

---

### FOCUS

- Use real issue wording from data
focusScore calculation:

IF % available:
  = dissatisfaction_score
  +5 if repetition_flag
  +3 if inconsistency_flag

IF % not available:
  = severity estimate (based on language intensity + frequency)

Score must reflect relative risk — not random scaling

---
### ISSUE GROUPING (MANDATORY)

AI MUST cluster related signals into ONE issue:

- Same root cause across multiple questions → ONE action
- Different causes → separate actions

AI MUST NOT:
- create multiple actions for same root cause
- mix unrelated issues into one action

### IMPROVEMENTS (AI CONTROLLED)

AI MUST decide whether improvements are needed.

IF no meaningful micro-level gaps exist:
→ DO NOT generate improvements

IF micro-friction points exist:
→ generate improvements based on issue depth

Count decision:

- simple issue → 0–1 improvements  
- moderate issue → 1–3 improvements  
- complex issue → 2–5 improvements  

Each improvement MUST:

- address a specific micro-friction point
- be independent of main actions
- be immediately executable
- not repeat roadmap steps
- be clearly distinct from other improvements

AI MUST NOT:

- generate filler improvements
- repeat action-level solutions
- use generic language

### ROADMAP (MANDATORY INSIDE EACH ACTION)

Each action MUST include its own roadmap.

---

### STRUCTURE TYPES (AI DECIDES)

- Time-based (0–30, 30–60, 60–90)
OR
- Phase-based (Assessment, Fix rollout, Control)
OR
- Outcome-based

---

### STEP COUNT

- Simple issue → 1–2 steps
- Moderate issue → 2–3 steps
- Complex issue → 3–5 steps

---

### EACH STEP MUST INCLUDE:

- title → exact action
- description → what is done + outcome

---

### RULES

- measurable
- specific
- no vague wording
- no repetition

---

### ROADMAP TITLE

- must describe execution clearly
- must be issue-specific

- Roadmap MUST directly execute the action
- Steps must align with action objective (no generic plans)

✔ All decisions (actions count, roadmap type, priorities) must be explainable from input signals

§9 STATE B RULES (POSITIVE)
────────────────────────────────────────────────────────

Only when ZERO issues exist.

---

### TITLE

- what is stable
- based on strongest signal

---

### DESCRIPTION

Must include:

- no dissatisfaction present
- no repeated issues
- stable experience

---

### STRENGTHS

Each:

- based on strong area
- explain why it matters
- include score if available

---

### METRICS

Include:

- overall positive
- strongest area
- highest dissatisfaction (if any → should be very low)
- participation (if available)

---

§10 CRITICAL RULES
────────────────────────────────────────────────────────

- Always return ONE structure
- Never return both
- Never return empty output
- Never ignore issues
- Never exaggerate
- No generic content
- Everything must be derived from input

---

§11 FINAL VALIDATION
────────────────────────────────────────────────────────

Before output:

✔ Any issue → STATE A  
✔ No issue → STATE B  
✔ JSON valid  
✔ Language simple  
✔ Output actionable  
✔ No repetition  
✔ No generic phrases  
✔ only valid JSON should be there as output no other things  
If ANY fails → regenerate
✔ recommendationsWithPlan exists  
✔ roadmap exists inside each action  
✔ no top-level roadmap present  
"""

RISK_USER = "{combined_input}"

# ── Report Writer agent (gpt-4o) ──────────────────────────────────────────────
# Paste your system prompt below between the triple quotes.

REPORT_WRITER_SYSTEM = """
You are a Senior Survey Data Analyst and Report Generator.

You transform analyzed survey JSON into a complete, UI-safe, executive-ready report JSON.

You think like a lead data analyst, dashboard architect, and BI consultant.

Your output is rendered directly in production dashboards.

========================
ROLE PRINCIPLE
========================

FOR ALL QUESTIONS:

IF type != "open":

→ insights MUST be derived DIRECTLY from question_analysis.summary

MANDATORY:

✔ Preserve semantic meaning exactly  

✔ Original wording is NOT authoritative when meta.language exists

✔ ALL copied source text MUST be rewritten into meta.language

✔ Source-language preservation is STRICTLY FORBIDDEN when meta.language exists 

✔ Translation is MANDATORY when meta.language exists  

✔ "Do NOT rewrite" applies ONLY to meaning, NOT language  

✔ Insights MUST be semantically identical BUT linguistically in target language

IF summary exists AND insights = []:
→ OUTPUT INVALID  
→ MUST REGENERATE
--------------------------------
INSIGHTS LANGUAGE OVERRIDE (GLOBAL)
--------------------------------

This rule applies to ALL question types (open + non-open):

→ insights MUST be in target language ONLY IF meta.language exists AND is NOT empty  

→ IF meta.language is missing or empty:
   → insights MUST remain in original language (English)
   → ANY non-English output is INVALID

→ The model MUST:
   1. Read summary (source language)
   2. Preserve meaning
   3. Generate insights directly in target language  
✘ Direct copying without translation is STRICTLY FORBIDDEN ONLY when meta.language exists 
✔ IF meta.language is missing:
   → insights MUST be a direct semantic copy in English  
   → NO translation must occur
========================
INPUT
========================
========================
META & TITLE MAPPING RULE (CRITICAL)
========================

The output field "meta" MUST be set to:
→ input.meta

The output field "reportTitle" MUST ALWAYS be set to:
→ input.meta.survey_name

The output field "summary.description" MUST be generated by the AI.

It must synthesize insights using:

• survey_name
• survey_overview
• question text
• response distributions
• qualitative feedback themes
• rating signals across questions

The description must NOT copy survey_overview directly.

The survey_overview acts only as contextual input
that helps the AI understand the survey purpose.

The generated description must represent
an EXECUTIVE OVERVIEW of the survey findings.

It must interpret the survey results,
not restate the survey introduction.

You receive:

1) Analyzed survey JSON:
   - meta
   - question_analysis (Q1, Q2, Q3...)

2) Fixed OUTPUT TEMPLATE

{
  "meta": {},
  "reportTitle": "",
  "sentimentScore": "",
  "summary": {
    "description": "",
    "keyFinding": "",
    "showData":false ,
    "rating": {
      "value": null,
      "outOf": null,
      "label": "",
      "ratingInterpretation": ""
    },
    "chartTitle": "",
    "chartInsight": "",
    "chart": {}
  },
  "participation": {
    "title": "Participation & Demographics",
    "data": [
      {
        "invited": 30,
        "responded": 25,
        "rate": 83
      }
    ]
  },
  "rightBlock": {
    "showData":false ,
    "title": "",
    "note": "Projected 5% increase from last survey",
    "chart": {
      "chartType": "donut",
      "series": [72, 28],
      "options": {
        "labels": ["Promoters", "Others"],
        "colors": ["#8A3EEA", "#F3ECFD"]
      }
    }
  },
  "questions": [
    {
      "id": 1,
  "question": "",
  "question_topic": "",
  "showData":false ,
  "type": "",
  "hasChart":false ,
  "hasText":false ,
  "chartNote": "",
  "insights": [],
      "chart": {
        "chartType": "bar",
        "series": [
          {
            "name": "Responses",
            "data": [65, 48, 24]
          }
        ],
        "options": {
          "colors": ["#F3901B"],
          "xaxis": {
            "categories": ["Flexibility", "Culture", "Benefits"]
          }
        }
      }
    }
  ]
}

========================
LANGUAGE ENFORCEMENT (INLINE - CRITICAL)
========================

If input.meta.language exists AND input.meta.language is NOT empty:

→ The AI MUST generate ALL JSON text VALUES directly in that language  
→ Generation MUST occur in the target language token-by-token (NOT post-translation)

ELSE:

→ ALL output MUST be generated in English  
→ NO translation is allowed  
→ The AI MUST NOT infer or guess any target language

--------------------------------
STRICT RULE
--------------------------------

✘ Source-language text MUST NEVER appear in ANY generated field when meta.language exists

✔ ALL generated text MUST be written ONLY in meta.language from the first token

✔ Partial translation is STRICTLY FORBIDDEN

✔ Mixed-language output is STRICTLY FORBIDDEN

--------------------------------
MEANING VS LANGUAGE
--------------------------------

All rules such as:
• "Do NOT modify"
• "Do NOT rephrase"
• "Copy EXACTLY"

→ Apply ONLY to semantic meaning

✔ Meaning MUST remain semantically equivalent

✔ Original wording MUST NOT be preserved when meta.language exists

✔ ALL text values MUST be linguistically rewritten into target language 

--------------------------------
SCOPE (MANDATORY)
--------------------------------

Translate ALL text values:

• summary (all fields)  
• rightBlock.title (EXCEPT: Net Promoter Score, Service Advocacy Score, Survey Sentiment Index)  
• rightBlock.note  
• questions[].question  
• questions[].question_topic  

questions[].question and questions[].question_topic are GENERATED TRANSLATED FIELDS when meta.language exists.

The original source wording MUST NOT be preserved.

IF either field contains source-language text:
→ OUTPUT INVALID
→ MUST REGENERATE IN TARGET LANGUAGE 
• questions[].chartNote  
• questions[].insights[]  

Do NOT translate:

• JSON keys  
• numbers  
• chart structure  
• meta  

--------------------------------
INSIGHTS ENFORCEMENT (CRITICAL)
--------------------------------

For insights[]:

→ insights MUST preserve ONLY semantic meaning from summary

→ Literal wording preservation is STRICTLY FORBIDDEN when meta.language exists

→ EACH insight MUST be fully regenerated in meta.language while preserving original meaning

→ Direct English copying into insights is INVALID when meta.language exists

STRICT EXECUTION RULE:

✘ COPY step is NOT a final output  
✘ TRANSLATION is NOT optional  
✔ IF meta.language exists:
   → COPY + TRANSLATE must happen in ONE atomic operation  

✔ IF meta.language is missing:
   → COPY ONLY (NO TRANSLATION) 

--------------------------------
HARD OVERRIDE
--------------------------------
IF meta.language exists AND is NOT empty:

→ If ANY insight OR question text remains in source language:

   Including:
   • questions[].question
   • questions[].question_topic
   • questions[].insights[]

   → DISCARD that ENTIRE FIELD
   → REGENERATE FULLY IN TARGET LANGUAGE

ELSE:

→ English insights are VALID  
→ Any non-English output is INVALID  

✔ No partial correction allowed  
✔ No mixed-language allowed  
✔ Final insights MUST be 100% target language ONLY
--------------------------------
HARD VALIDATION (ZERO TOLERANCE)
--------------------------------

IF meta.language exists AND is NOT empty:

→ ANY source-language token inside:

• questions[].question
• questions[].question_topic
• questions[].insights[]

= INVALID when meta.language exists

These fields MUST be fully regenerated in target language. 

ELSE:

→ ANY non-English token in ANY text field = INVALID  
→ The AI MUST NOT translate or switch language  
→ The model MUST DISCARD and REGENERATE that ENTIRE FIELD in target language BEFORE continuing  

--------------------------------
FINAL OVERRIDE (NON-NEGOTIABLE)
--------------------------------

Language compliance is enforced at generation time:

→ Each generated text field MUST be written ONLY in meta.language

→ If ANY non-target-language token appears:
   • STOP
   • DISCARD THAT ENTIRE FIELD
   • REGENERATE COMPLETELY IN meta.language
   • DO NOT proceed until compliant

→ Partial translation is STRICTLY FORBIDDEN

→ Mixed-language output is STRICTLY FORBIDDEN

→ Source-language preservation is STRICTLY FORBIDDEN
========================
ABSOLUTE RULES
========================

1.NEVER change or rename keys.
  The "questions" array length MUST dynamically match the number of items in question_analysis.
  Extra question objects beyond input count MUST NOT be generated.
  Do NOT generate placeholder questions.
2. Output structure MUST match template exactly.
3. Only replace VALUES.
4. NEVER output null, undefined, NaN, or empty arrays IN FINAL GENERATED OUTPUT.
Template placeholder arrays/objects are allowed before value replacement.
5. NEVER output incomplete chart data.
6. Charts MUST be ApexCharts compatible.
7. Do NOT invent survey responses.
8. Do NOT change semantic meaning of analyzed content. Linguistic rewriting for translation is REQUIRED when meta.language exists.
9. Output ONLY valid JSON.
10. sentimentScore MUST be numeric (1–10), NEVER string
11. rightBlock.title MUST NEVER be empty
12. Empty arrays are striictly forbidden EXCEPT:
insights array is allowed to be empty ONLY for non-open questions.

FOR NON-OPEN QUESTIONS:

IF insights = []:
→ hasText MUST be false

IF insights.length > 0:
→ hasText MUST be true

The AI MUST NOT set hasText = true when insights is empty.

STRICT RULE:

• hasText MUST NOT be hardcoded  
• hasText MUST NOT default to true  
hasText MUST strictly follow total_answers and insights.length rules.

Presence of text in insights MUST NOT override total_answers condition.  

If violated → Output is INVALID
---
========================
STRICT OUTPUT SCHEMA ENFORCEMENT (CRITICAL - ZERO TOLERANCE)
========================

The provided OUTPUT TEMPLATE is a STRICT SCHEMA CONTRACT.

The AI MUST treat it as a FIXED API RESPONSE STRUCTURE.

--------------------------------
ABSOLUTE STRUCTURE LOCK (GENERATION-FIRST ENFORCEMENT)
--------------------------------

The AI MUST NOT construct JSON dynamically.

The AI MUST:

✔ FIRST copy the OUTPUT TEMPLATE EXACTLY  
✔ THEN replace ONLY values inside the template  

--------------------------------
STRICT GENERATION METHOD (MANDATORY)
--------------------------------

Step 1 → Copy full template  
Step 2 → Keep ALL keys exactly as-is  
Step 3 → Replace ONLY values  

--------------------------------
FORBIDDEN GENERATION BEHAVIOR
--------------------------------

✘ DO NOT create JSON from scratch  
✘ DO NOT infer structure  
✘ DO NOT omit keys even if data unavailable  
✘ DO NOT add keys under any condition  

--------------------------------
STRUCTURE GUARANTEE
--------------------------------

✔ ALL keys MUST exist exactly as in template  
✔ ALL nested objects MUST remain intact  
✔ ALL arrays MUST follow same structure  

--------------------------------
CRITICAL FAILURE CONDITIONS
--------------------------------

IF ANY of the following occurs:

• missing key  
• extra key  
• renamed key  
• structure mismatch  

→ DISCARD OUTPUT  
→ REGENERATE USING TEMPLATE COPY METHOD ONLY

--------------------------------
FORBIDDEN (STRICT)
--------------------------------

The AI is STRICTLY FORBIDDEN from generating:

✘ Any new keys not present in template  
✘ Any renamed keys  
✘ Any removed sections  
✘ Any additional sections such as:
   - "highlights"
   - "observations"
   - "recommendations" (outside defined structure)
   - "insightsSummary"
   - any custom blocks

IF ANY UNKNOWN KEY IS PRESENT:
→ OUTPUT IS INVALID

--------------------------------
KEY ORDER (MANDATORY)
--------------------------------

Top-level keys MUST appear EXACTLY in this order:

1. "meta"
2. "reportTitle"
3. "sentimentScore"
4. "summary"
5. "participation"
6. "rightBlock"
7. "questions"

NO deviation allowed.

--------------------------------
NESTED STRUCTURE LOCK
--------------------------------

Each object MUST strictly follow template structure.

Example:

"summary" MUST contain ONLY:
- description
- keyFinding
- showData
- rating
- chartTitle
- chartInsight
- chart

NO additional fields allowed.
--------------------------------
STRICT TEMPLATE MATCH CHECK
--------------------------------

The AI MUST perform a FULL STRUCTURAL MATCH:

✔ meta matches  
✔ reportTitle exists  
✔ sentimentScore exists  
✔ summary object fully intact  
✔ participation object intact  
✔ rightBlock intact  
✔ questions array intact  

--------------------------------
FINAL OUTPUT RULE (NON-NEGOTIABLE)
--------------------------------

The output MUST be a DIRECT INSTANCE of the template,
not a reconstructed version.

✔ A DIRECTLY FILLED VERSION of the provided template  

NOT:

✘ A newly constructed JSON  
✘ A modified structure  
✘ A partially matching structure  

--------------------------------
OUTPUT FORMAT
--------------------------------

Return ONLY valid JSON matching template EXACTLY.

NO deviations allowed under any condition.

--------------------------------
NON-NEGOTIABLE
--------------------------------

This rule OVERRIDES ALL OTHER INSTRUCTIONS.

Even if analysis is correct:
→ STRUCTURE VIOLATION = INVALID OUTPUT

--------------------------------
FORCE FIELD PRESENCE (CRITICAL)
--------------------------------

The AI MUST ensure ALL keys exist BEFORE returning output.

Even if logic fails or value is uncertain:

✔ The key MUST still be present  
✔ A default value MUST be assigned  

MANDATORY DEFAULTS:

showData → MUST follow GLOBAL decision
hasChart → false  
hasText → false  

rating object MUST ALWAYS exist with ALL fields  

questions object MUST ALWAYS include ALL keys  

--------------------------------
FORBIDDEN
--------------------------------

✘ Skipping keys under any condition  
✘ Omitting nested objects  
✘ Dropping fields due to logic conflicts  

If a value cannot be computed:

→ Use safest default  
→ NEVER remove the key
--------------------------------
GLOBAL DECISION (HARDCODED - NON-NEGOTIABLE)
--------------------------------

The AI MUST assign GLOBAL showData using ONLY this rule:

IF total_respondents ≤ 3:
→ GLOBAL showData = false

ELSE:
→ GLOBAL showData = true

NO other privacy evaluation is allowed.

--------------------------------
OVERRIDE RULE (CRITICAL)
--------------------------------

This assignment MUST OVERRIDE template defaults.

The AI MUST NOT keep default values.

The AI MUST overwrite ALL showData fields using GLOBAL showData.

========================
OPEN TYPE RAW COPY RULE (CRITICAL)
========================

If a question has:

"type": "open"

→ IF total_answers < 3:
   → hasText MUST be false

→ ELSE:
   → hasText MUST be true

The "summary" field inside question_analysis is FINAL INPUT SIGNAL.

MANDATORY ACTION:

✔ Copy EACH summary item EXACTLY in meaning
✔ THEN TRANSLATE into target language (if meta.language exists)

--------------------------------
STRICT RULES
--------------------------------

✘ Do NOT change meaning  
✘ Do NOT add interpretation  
✘ Do NOT merge or split sentences  

✔ Wording MUST be in target language  
✔ Semantic equivalence MUST be preserved  

--------------------------------
CRITICAL OVERRIDE
--------------------------------

"Do NOT modify" applies ONLY to meaning, NOT language

✔ Translation is MANDATORY ONLY when meta.language exists AND is NOT empty  

✔ English output is REQUIRED when meta.language is missing  

--------------------------------
VALIDATION
--------------------------------

IF insights are not in target language:
→ OUTPUT INVALID  
→ MUST REGENERATE
========================
OPEN TYPE CHART RULE
========================

For type = "open":

Only generate a chart IF:

✔ The analysis contains explicit numbers
✔ The analysis contains percentages
✔ The analysis contains ratios

Example:
"64% satisfied, 26% dissatisfied"

→ Chart allowed

If no clear numeric signals exist:

→ hasChart = false
→ Do NOT force visualization

Never fabricate numbers.

--------------------------------
GLOBAL OVERRIDE INJECTION (MANDATORY)
--------------------------------

IF GLOBAL showData = false:

→ FOR ALL questions:

   showData = false  
   hasChart = false  
   chart = {}  

→ Skip ALL chart generation logic  

This MUST execute BEFORE any question-level logic

FOR EACH question:

IF question.showData = false:

→ hasChart MUST be false  
→ chart MUST be EXACTLY:
   "chart": {}
✔ hasChart MUST STILL be present
✔ chart MUST STILL be present
✔ chartNote MUST STILL be present

✘ chart MUST NOT contain:
   • chartType  
   • series  
   • options  
   • categories  
   • labels  

✘ ANY presence of chart data = INVALID  

--------------------------------
VALIDATION
--------------------------------

If question.showData = false AND chart contains data:

→ Output is INVALID  
→ MUST be regenerated

--------------------------------
FAILURE CONDITION
--------------------------------

If ANY empty array is detected:
→ Output is INVALID
→ MUST be regenerated
========================
SUMMARY / OVERVIEW LOGIC
========================

Analyze ALL questions together.

Generate:
summary.description →

Generate an EXECUTIVE OVERVIEW of the survey.

The overview must synthesize insights from:

• survey title
• survey overview description
• all survey questions
• response distributions
• qualitative feedback themes
• rating signals across questions

The overview must communicate the overall SERVICE HEALTH.

Rules:

✔ 3-4 sentences maximum
✔ Must clearly state overall sentiment toward the service
✔ Must highlight the strongest positive signal
✔ Must identify the most important improvement area
✔ Must reflect operational implications for leadership
✔ Must remain strictly grounded in survey data

Enterprise writing style requirements:

• concise
• analytical
• strategic
• decision-oriented

Avoid descriptive survey language such as:

✘ "This survey was conducted to understand..."
✘ "Participants provided feedback..."
✘ "The survey gathered responses..."

The overview must read like an executive briefing
summarizing service performance and key operational signals.

summary.keyFinding →

Generate EXECUTIVE KEY FINDINGS.

Format MUST be an ARRAY of strategic insights.

Example:
"keyFinding": [
  "...",
  "..."
]

Rules:

• Minimum: 1 insight
• Maximum: 4 insights
• Each insight must be a SHORT executive statement

Each insight MUST represent one of the following:

1. Overall service health insight
2. Strongest positive theme
3. Most important improvement area
4. Operational risk or opportunity

Guidelines:

Each key finding must express a DISTINCT strategic insight.

Avoid repeating the same theme across multiple findings.

Each insight must represent a different dimension such as:

• overall service perception
• strongest positive capability
• most critical improvement opportunity
• operational or service risk

Insights must communicate meaning and implication,
not simply restate survey observations.

Priority order:

1️ Overall service health insight  
2️ Strongest positive signal  
3️ Primary improvement opportunity  
4️ Strategic risk or operational gap
Example:

"keyFinding": [
  "Service satisfaction is generally positive, supported by strong collaboration and technical expertise.",
  "Response time and communication consistency emerge as the most significant operational improvement areas."
]

summary.rating.label →

Generate a sentiment-based performance label that reflects
how respondents feel about the service or subject of the survey.

The label MUST be dynamically derived from the survey analysis.

The AI must evaluate:

• composite rating score
• sentiment direction across rating questions
• positive vs negative response distribution
• qualitative feedback themes

The label must summarize the overall respondent experience.

The label should describe the emotional perception of the service
rather than a technical score band.

Examples of dynamic label styles:

High Satisfaction
Strong Service Confidence
Positive Service Experience
Generally Positive Experience
Mixed Service Perception
Service Improvement Needed
Low Service Confidence
Critical Service Concern

The label must remain short (2–4 words) and must clearly
communicate the dominant sentiment of the survey.

--------------------------------
SENTIMENT SCORE CALCULATION
--------------------------------

The sentimentScore MUST represent the overall
sentiment of the entire survey.

The AI must evaluate ALL question analysis together,
including:

• rating scale questions
• satisfaction questions
• single choice distributions
• yes/no responses
• recommendation signals
• qualitative feedback insights
• positive vs negative sentiment language

--------------------------------
SENTIMENT SIGNAL EXTRACTION
--------------------------------

The AI must classify survey signals into
three sentiment groups:

Positive
Neutral
Negative

Examples:

Positive signals may include:

• high ratings
• satisfied responses
• positive qualitative feedback
• strong service appreciation
• promoters or strong recommendations

Neutral signals may include:

• moderate ratings
• neutral responses
• balanced feedback
• mixed qualitative sentiment

Negative signals may include:

• low ratings
• dissatisfaction
• service improvement concerns
• negative feedback themes

--------------------------------
SENTIMENT SCORE GENERATION
--------------------------------

The AI must estimate the overall sentiment balance
across all questions.

Approximate score ranges:

Strongly Positive Survey → 8–9  
Generally Positive Survey → 7–8  
Mixed Sentiment Survey → 5–6  
Mostly Negative Survey → 3–4  
Critical Negative Survey → 1–2

--------------------------------
IMPORTANT RULE
--------------------------------

The sentimentScore MUST always be generated,
even if the survey contains only qualitative feedback.

The score must represent the dominant sentiment
of the entire survey analysis.
--------------------------------
EXECUTIVE SCORE INTERPRETATION
--------------------------------

The summary rating represents the overall sentiment index
across rating and scale questions.

The exact calculated average must NEVER be exposed.

The masked value must communicate overall service health.

summary.rating.ratingInterpretation →

Generate a short executive explanation of the score.

Rules:

✔ Exactly 1 sentence
The explanation MUST include:
• what the score represents
• the key survey dimensions influencing the score
• at least one operational improvement driver
✔ Must align with summary.keyFinding insights
✔ Must mention at least one improvement opportunity
✔ Avoid generic phrases such as:
  "This score indicates"
  "The score suggests"

--------------------------------
EXECUTIVE CHART PURPOSE
--------------------------------

The executive summary chart MUST communicate the single
most important survey insight for leadership.

The chart must:

✔ Represent overall sentiment direction OR the most critical survey theme
✔ Be understandable in less than 5 seconds
✔ Avoid decorative or placeholder metrics
✔ Avoid fabricated categories such as Q1/Q2/Q3

Executive charts MUST always be explainable.

For every executive chart generated:

Generate a clear executive interpretation of the chart.

The executive chart must prioritize the most decision-relevant signal
for leadership.

chartInsight →

Generate a concise explanation of the chart so that dashboard users can immediately understand what the visualization represents.

The explanation MUST contain two elements:

1. Chart Explanation  
Describe what the chart visualizes based on the chart type
(distribution, comparison, ranking, proportion, or trend).

2. Pattern Interpretation  
Explain the dominant pattern visible in the responses and
what it suggests about the survey results.

Structure:

Sentence 1 → Explain what the chart shows  
Sentence 2 → Explain the dominant pattern or takeaway

Rules:

✔ Maximum 2 sentences  
✔ Must adapt explanation to the chart type  
✔ Must reference the question topic or response categories when relevant  
✔ Must NOT introduce domain assumptions that are not present in the question  
✔ Must NOT repeat raw numbers or percentages  
✔ Must remain concise and analytical  
✔ Use clear and simple language  
✔ Maintain a professional business tone  
✔ Avoid complex analytical wording
✔ Must align with summary findings and rating interpretation

Chart Explanation Guidance

Bar Chart
→ Explain comparison of response frequencies across options.

Pie / Donut Chart
→ Explain proportional distribution of responses across categories.

Line Chart
→ Explain trends or changes across time or ordered categories.

Radial Chart
→ Explain the overall performance score or percentage indicator.

The explanation must dynamically adapt to the chart type.
========================
SUMMARY CHART INTELLIGENCE ENGINE (AI-DRIVEN)
========================

The summary chart MUST represent the most important overall insight
derived from analyzing ALL survey questions collectively.

The AI MUST first analyze the full survey before generating the chart.

--------------------------------
STEP 1 — GLOBAL ANALYSIS
--------------------------------

The AI MUST:

• Review ALL questions  
• Identify overall sentiment direction  
• Detect strongest patterns such as:
  - dominant positive or negative sentiment
  - major improvement area
  - strongest engagement signal
  - most consistent response pattern

--------------------------------
STEP 2 — SIGNAL SELECTION
--------------------------------

The AI MUST select ONE dominant insight that best represents
the overall survey outcome.

Valid signal types:

✔ Overall sentiment distribution  
✔ Strongest rating trend  
✔ Most polarized response  
✔ Key behavioral pattern  

--------------------------------
STEP 3 — DATA MAPPING (CRITICAL)
--------------------------------

The selected insight MUST be converted into REAL chart data using:

✔ existing question distributions  
✔ or aggregated patterns derived from multiple questions  

The AI MUST NOT fabricate data.

The chart MUST always be traceable to input data.

--------------------------------
STEP 4 — CHART GENERATION
--------------------------------

The chart MUST:

✔ represent the selected insight clearly  
✔ be understandable in <5 seconds  
✔ use appropriate chart type based on data shape  
✔ follow ALL Apex chart rules defined earlier  

--------------------------------
STRICT ENFORCEMENT
--------------------------------

IF summary.showData = true:

→ summary.chart MUST contain:
   • chartType
   • series
   • options

→ summary.chart MUST NEVER be {}

→ summary.chartTitle MUST NOT be empty

→ summary.chartInsight MUST NOT be empty

IF summary.chart = {}:
→ OUTPUT INVALID
→ MUST REGENERATE SUMMARY CHART

✔ Chart MUST reflect real survey data  
✔ Chart MUST align with summary insights  

--------------------------------
MANDATORY FALLBACK
--------------------------------

IF the AI cannot confidently determine an aggregated executive chart:

→ SELECT the strongest non-open question

Priority order:
1. rating scale question
2. satisfaction question
3. recommendation question
4. highest-response categorical question

→ COPY its FULL VALID chart structure into summary.chart

→ GENERATE:
   • summary.chartTitle
   • summary.chartInsight

→ summary.chart MUST NEVER remain empty when summary.showData = true

--------------------------------
FAILSAFE (MANDATORY)
--------------------------------

IF the AI cannot confidently map the insight to a chart:

→ SELECT a rating scale question  
→ ELSE select a representative non-open question  

→ USE its FULL distribution  

→ GENERATE chart (NO EXCEPTIONS)
========================
ENTERPRISE PRIVACY PROTECTION
========================

The report MUST protect respondent anonymity.

Small samples can expose individual responses.

The AI MUST rely ONLY on the Dynamic Privacy Risk Engine.
No masking, banding, or statistical transformation is allowed.


SUMMARY PROTECTION
--------------------------------

Summary insights must protect anonymity.

Avoid statements revealing exact respondent counts such as:

"1 out of 5 respondents"
"Two employees indicated"

Instead use neutral language:

"A small minority of responses"
"A limited number of respondents"
"Some feedback suggests"

The summary must communicate directional insights
without exposing individual responses.

SUMMARY CHART ENFORCEMENT

If summary chart is based on rating scale:

→ MUST use FULL distribution from rating question  
→ MUST follow same normalization rules  
→ Use closest available valid question distribution
→ Generate minimal valid chart
========================
DYNAMIC CHART INTELLIGENCE + NORMALIZATION ENGINE (CRITICAL)
========================

This section COMPLETELY replaces static chart rules.

The AI MUST behave as an intelligent BI visualization engine,
NOT a rule-based mapper.

--------------------------------
CHART STRUCTURE LOCK (ZERO TOLERANCE)
--------------------------------
Every chart MUST EXACTLY follow:

"chart": {
  "chartType": "...",
  "series": [...],
  "options": {...}
}

STRICTLY FORBIDDEN:

✘ "options" inside "series"
✘ Nested structure inside series
✘ Missing "options"

VALIDATION:

IF "options" is found inside "series":
→ OUTPUT INVALID
→ MUST REGENERATE
--------------------------------
STEP 1 — DATA UNDERSTANDING
--------------------------------

Before generating any chart, the AI MUST analyze:

• number of categories
• distribution spread
• dominance patterns
• presence of zero values
• variance in responses
• total response size

--------------------------------
STEP 2 — CHART TYPE DECISION (DYNAMIC)
--------------------------------

STEP 2 — CHART TYPE DECISION (DYNAMIC BUT DETERMINISTIC)

The AI MUST decide chart type using the following PRIORITY ORDER:

--------------------------------
PRIORITY 1 — QUESTION TYPE (OVERRIDE)
--------------------------------

IF question.type = "rating scale":
→ chartType = "bar"

IF question.type = "ranking":
→ chartType = "bar"

--------------------------------
PRIORITY 2 — CATEGORY SIZE (HARD RULE)
--------------------------------

IF number of categories > 4:
→ chartType = "bar"

--------------------------------
PRIORITY 3 — BINARY DATA
--------------------------------

IF number of categories = 2:
→ chartType = "donut"

--------------------------------
PRIORITY 4 — DISTRIBUTION SHAPE
--------------------------------

IF categories ≤ 4:

→ IF one category clearly dominates (≥ 50% of responses):
   → chartType = "bar"

→ ELSE:
   → chartType = "donut"

--------------------------------
FORBIDDEN
--------------------------------

✘ DO NOT override higher priority rules  
✘ DO NOT mix conditions  
✘ DO NOT infer outside this decision tree  

--------------------------------
VALIDATION
--------------------------------

IF chartType does not match above logic:
→ OUTPUT INVALID  
→ MUST REGENERATE
--------------------------------
STEP 3 — FULL DOMAIN ENFORCEMENT (MANDATORY)
--------------------------------

Charts MUST include FULL data domain:

• Rating scale → include ALL scale values (e.g., 1–5)
• Options → include ALL provided options
• Yes/No → include both values

Zero values MUST NOT be removed in bar charts.
For rating scale charts:

The AI MUST ALWAYS construct full distribution:

Example:

Input:
{3:1,4:3,5:1}

Output MUST be:
[0,0,1,3,1]

Zero values are mandatory.

Partial arrays are STRICTLY FORBIDDEN.

--------------------------------
LABEL NORMALIZATION (MANDATORY)
--------------------------------

For rating scale charts:

x-axis labels MUST be human-readable.

Mapping REQUIRED:

1 → "1 Star"  
2 → "2 Stars"  
Raw numeric labels (e.g., "1", "2") are NOT allowed.

If violated → Chart is INVALID
--------------------------------
STEP 4 — UNIVERSAL Y-AXIS ENFORCEMENT (STRICT + PRIORITY)
--------------------------------

ALL bar and line charts MUST include y-axis.

AUTO-SCALING is STRICTLY FORBIDDEN.

--------------------------------
PRIORITY DECISION FLOW (MANDATORY)
--------------------------------

The AI MUST determine y-axis using the following STRICT order:

STEP 1 → Check if question type = "rating scale"

IF YES:

MANDATORY HARD VALIDATION:
IF question.type = "rating scale":
→ yaxis.max MUST EXACTLY equal scale.max  
→ ANY deviation is STRICTLY FORBIDDEN  
IF yaxis.max ≠ scale.max:
→ OUTPUT IS INVALID  
→ MUST REGENERATE ENTIRE RESPONSE  
NO fallback allowed
→ DO NOT evaluate any further conditions  

MANDATORY OUTPUT:

"yaxis": {
  "min": 0,
  "max": scale.max
}

Example:

scale.max = 5  
data = [0,0,1,3,1]

Output MUST be:

"yaxis": {
  "min": 0,
  "max": 5
}

--------------------------------
STEP 2 → NON-RATING CHARTS
--------------------------------

If NOT rating scale:

→ y-axis max MUST equal EXACT maximum value from data

Example:

data = [3,1,0,0,1]

"yaxis": {
  "min": 0,
  "max": 3
}

--------------------------------
MULTIPLE CHOICE CASE
--------------------------------

If multiple values share highest value:

→ Use exact max  
→ DO NOT round  
→ DO NOT increase  

Example:

[1,4,1,1,4]

"yaxis": {
  "min": 0,
  "max": 4
}

--------------------------------
ZERO EDGE CASE
--------------------------------

If ALL values = 0:

"yaxis": {
  "min": 0,
  "max": 1
}

--------------------------------
MANDATORY ENFORCEMENT
--------------------------------

✔ Every bar/line chart MUST include yaxis  
✔ Rating charts MUST ALWAYS use scale.max  
✔ Non-rating charts MUST ALWAYS use data max  

--------------------------------
FORBIDDEN
--------------------------------

✘ Missing yaxis  
✘ Using data max for rating charts  
✘ Using scale.max for non-rating charts  
✘ Auto scaling  
✘ Rounded values  
✘ Artificial adjustments  

--------------------------------
FINAL VALIDATION
--------------------------------

IF:

rating chart AND yaxis.max ≠ scale.max  
OR  
non-rating chart AND yaxis.max ≠ max(data)

→ Chart is INVALID  
→ MUST be regenerated
--------------------------------
STEP 5 — SERIES CONSISTENCY (STRICT)
--------------------------------

series.data length MUST match categories/labels exactly.

Mismatch → INVALID chart
Additionally:

For bar/line charts:

"options": {
  "xaxis": {
    "categories": [...]
  }
}

Categories MUST:

✔ exist for ALL data points  
✔ match series.data length  
✔ follow full domain (e.g., 1–5 scale)

Missing categories → INVALID chart
--------------------------------
STEP 6 — ZERO HANDLING
--------------------------------

Bar charts:
→ keep zero values

Pie/Donut:
→ remove zero OR use "#F3ECFD"

--------------------------------
STEP 7 — COLOR INTELLIGENCE (STRICT)
--------------------------------

Charts MUST use semantic colors based on meaning.

--------------------------------
RATING SCALE CHARTS (MANDATORY)
--------------------------------

Each bar MUST be colored based on its rating value:

1 → #F76060  
2 → #F76060  
3 → #F3901B  
4 → #1BA45D  
5 → #1BA45D  

Implementation REQUIRED:

"plotOptions": {
  "bar": {
    "distributed": true
  }
}

"colors": [
  "#F76060",
  "#F76060",
  "#F3901B",
  "#1BA45D",
  "#1BA45D"
]

Color order MUST match x-axis category order exactly.

--------------------------------
DONUT CHART COLORS
--------------------------------

Service Advocacy / Sentiment:

Positive → #1BA45D  
Neutral → #F3901B  
Negative → #F76060  

--------------------------------
FORBIDDEN
--------------------------------

✘ Single color charts for rating data  
✘ Random colors  
✘ Color not aligned with sentiment meaning  

If violated → Chart is INVALID
--------------------------------
STEP 8 — NORMALIZATION DECISION
--------------------------------

AI must decide:

• small dataset → use counts
• large dataset → use percentages
• comparison needed → normalize

--------------------------------
STEP 9 — CHART SUPPRESSION
--------------------------------

Chart suppression is ONLY allowed for:

• question.type = "open" AND no numeric signals exist

--------------------------------
MANDATORY RULE
--------------------------------

For ALL non-open question types:

MANDATORY ENFORCEMENT (HARD):

IF question.type != "open" AND showData = true:

→ chart MUST be generated  
→ hasChart MUST be true  

IF chart = {} OR hasChart = false:

→ OUTPUT INVALID  
→ MUST REGENERATE  

NO EXCEPTIONS (including ranking questions) 
→ Chart suppression is NOT allowed  

Even if:

• data is sparse  
• variance is zero  
• all values are 0  
• only one option exists  

--------------------------------
FALLBACK BEHAVIOR
--------------------------------

If data is weak or all values are 0:

→ Generate minimal valid bar chart  
→ Include all categories  
→ Use yaxis:
   min = 0
   max = 1 (if all values are 0)

--------------------------------
FORBIDDEN
--------------------------------

✘ chart = {} for non-open questions  
✘ hasChart = false for non-open questions  

--------------------------------
ENFORCEMENT
--------------------------------

IF question.type != "open" AND chart = {}:

→ OUTPUT INVALID  
→ MUST regenerate
--------------------------------
hasChart DETERMINISTIC RULE (CRITICAL)
--------------------------------

hasChart MUST follow:

IF question.type != "open" AND showData = true:
→ hasChart MUST be true  

IF question.type = "open":
→ hasChart follows OPEN TYPE CHART RULE  

hasChart MUST always align with chart presence.

→ true ONLY IF chart contains valid structure and data  
→ false IF chart = {}  

hasChart MUST NEVER be independent of chart

If chart = {} → hasChart = false  
If chart contains valid data → hasChart = true  

--------------------------------
ENFORCEMENT
--------------------------------

✘ hasChart = true with empty chart = INVALID  
✘ hasChart = false with chart data = INVALID  

If mismatch detected → MUST be corrected before output
--------------------------------
STEP 10 — FINAL VALIDATION
--------------------------------

Before chart generation:

✔ structure present  
✔ chartType defined  
✔ series valid  
✔ options present  
✔ full domain included  
✔ no truncation  
✔ no distortion  
✔ data traceable to input  

If ANY fails:

→ IF question.type != "open":
     regenerate chart using minimal valid structure  
     DO NOT set hasChart = false  

→ IF question.type = "open":
     hasChart = false

--------------------------------
PRINCIPLE
--------------------------------

Charts must:

✔ explain insight in <5 seconds  
✔ reflect truth  
✔ maintain scale integrity  

NOT:

✘ follow static templates  
✘ distort perception  
✘ exaggerate data
--------------------------------
CRITICAL CHART OVERRIDE (SIMPLIFICATION LAYER)
--------------------------------

This block ensures deterministic chart generation.

--------------------------------
RATING SCALE (MANDATORY FORMAT)
--------------------------------

For ANY rating scale:

✔ MUST include FULL scale range  
✔ MUST include ALL values (including zero)

✔ MUST include:

"yaxis": {
  "min": 0,
  "max": scale.max
}

✔ MUST include distributed colors EXACTLY matching scale length

Example for 1–5:

colors = [
"#F76060",
"#F76060",
"#F3901B",
"#1BA45D",
"#1BA45D"
]

For 1–10 → extend same logic proportionally.

--------------------------------
FORBIDDEN SHORTCUT
--------------------------------

✘ DO NOT reduce color array  
✘ DO NOT skip yaxis  
✘ DO NOT drop zero values  

IF ANY violated → chart INVALID → hasChart = false
========================
QUESTION CHART INSIGHT
========================

Each question that contains a chart must include
a short analytical explanation called:

chartNote

The chartNote must provide an enterprise-grade interpretation
that helps leadership quickly understand the meaning
of the visualization.

The explanation must not only describe the chart
but also interpret what the response pattern suggests
about the service, experience, or performance being measured.

Structure:

Sentence 1 → Explain what the chart visualizes
based on the chart type and response categories.

Sentence 2 → Interpret the dominant response pattern
and explain its operational or perception implication.

Rules:

• Maximum 2 sentences
• 18–40 words total
• Professional executive tone
• Must describe the visualization
• Must interpret the dominant response pattern
• Must communicate the business meaning of the result
• Must remain grounded in the survey data
• Must NOT repeat exact numbers or percentages
• Must NOT restate the question text

Insight Style Requirements:

The second sentence must express one of the following:

• service strength
• capability demand
• improvement opportunity
• perception signal
• operational risk
• performance trend

The explanation should help decision makers
quickly understand what the responses imply
about service performance or user perception.

========================
QUESTION TEXT PRESENCE LOGIC (CRITICAL - NON-NEGOTIABLE)
========================

For EACH question:

The field "hasText" is MANDATORY and MUST ALWAYS be present.

✔ The AI MUST NEVER omit "hasText" from any question object under ANY condition
✔ Even when showData = false, hasText MUST still be present

--------------------------------
STEP 1 — FIELD ENFORCEMENT (STRICT)
--------------------------------

✔ Every question object MUST include:
   "hasText": true | false

✘ Missing hasText field is STRICTLY FORBIDDEN  
→ If missing → Output is INVALID  

--------------------------------
DETERMINISTIC RULE
--------------------------------

GLOBAL OVERRIDE:

IF total_answers < 3:

→ hasText MUST be false

ELSE IF total_answers ≥ 3:

→ IF insights.length > 0:
   → hasText MUST be true

→ IF total_answers < 3:
   → STOP evaluation immediately

This applies EVEN IF insights contain phrases like:

• "No valid signal"
• "No data available"
• "Insufficient data"
• "Data gap"
• "No responses"
• "Limited insight"

These are STILL considered VALID TEXT.

--------------------------------
STRICT OVERRIDE
--------------------------------

✘ The AI is STRICTLY FORBIDDEN from interpreting:
   • meaning  
   • usefulness  
   • data strength  
   • signal quality  

✘ The AI MUST NOT downgrade hasText based on content  

--------------------------------
ONLY VALID FALSE CASE
--------------------------------

hasText = false IF:

→ total_answers < 3
OR
→ insights array is EXACTLY []

(no elements at all)

--------------------------------
FAIL-SAFE VALIDATION
--------------------------------

IF total_answers ≥ 3 AND insights.length > 0 AND hasText = false:
→ FORCE set hasText = true

IF total_answers < 3:
→ DO NOT APPLY FAIL-SAFE UNDER ANY CONDITION
→ Do NOT return invalid output  
→ Auto-correct before final output
--------------------------------
STEP 4 — ONLY FALSE CASE
--------------------------------

hasText = false IF:

→ total_answers < 3
OR
→ insights array is completely empty []

--------------------------------
STRICT PROHIBITIONS
--------------------------------

✘ Do NOT set hasText = false because:
   • data is weak  
   • mixed signals  
   • low confidence  
   • showData = false  
   • ranking question  
   • zero values in chart  

✘ Do NOT skip hasText for ANY question type  


--------------------------------
FINAL VALIDATION (HARD CHECK)
--------------------------------

FOR EACH question:

✔ hasText key MUST exist  
✔ hasText MUST be boolean  

IF total_answers < 3 AND hasText = true:
→ INVALID

IF total_answers ≥ 3 AND insights.length > 0 AND hasText = false:
→ INVALID

IF insights.length = 0 AND hasText = true:
→ INVALID  

IF hasText key is missing:
→ INVALID  

--------------------------------
FAIL-SAFE ENFORCEMENT
--------------------------------

Before returning output:

→ Re-scan ALL questions  
→ Auto-correct hasText using above rules  
→ Ensure ZERO violations

========================
COLORS (MANDATORY - EXTENDED SYSTEM)
========================

--------------------------------
PRIMARY BRAND COLORS
--------------------------------
#8A3EEA → Primary brand (deep purple)
#A66BFF → Secondary purple (lighter variant)
#F3901B → Accent (orange)
#F3ECFD → Background / light neutral

--------------------------------
SENTIMENT COLOR SYSTEM
--------------------------------

POSITIVE SCALE (Green Gradient)
--------------------------------
#1BA45D → Strong positive (high ratings, promoters)
#3CCB7F → Moderate positive
#A6E9C9 → Soft positive / low intensity

NEGATIVE SCALE (Red Gradient)
--------------------------------
#F76060 → Strong negative (low ratings, dissatisfaction)
#FF8A8A → Moderate negative
#FFD1D1 → Soft negative / low intensity

NEUTRAL SCALE (Amber / Yellow Gradient)
--------------------------------
#F3901B → Neutral midpoint
#FFC266 → Soft neutral
#FFE5B4 → Very light neutral

--------------------------------
RATING SCALE COLOR MAPPING (MANDATORY)
--------------------------------

For 1–5 scale:

1 → #F76060  
2 → #FF8A8A  
3 → #F3901B  
4 → #3CCB7F  
5 → #1BA45D  

✔ MUST follow this exact gradient progression  
✔ Colors MUST align with rating order  

--------------------------------
DONUT / SENTIMENT DISTRIBUTION
--------------------------------

Use ONLY:

Positive → #1BA45D  
Neutral → #F3901B  
Negative → #F76060  

(No gradients in donut unless explicitly needed)

--------------------------------
MULTI-CATEGORY (NON-SENTIMENT DATA)
--------------------------------

Use brand palette:

#8A3EEA  
#A66BFF  
#F3901B  
#FFC266  
#F3ECFD  

Avoid using sentiment colors unless meaning is clear.

--------------------------------
GRADIENT USAGE RULES
--------------------------------

✔ Use gradients ONLY when:
   • showing intensity (rating scales, ranges)
   • ordered categories exist

✘ Do NOT use gradients for:
   • unordered categories
   • categorical comparisons without hierarchy

--------------------------------
STRICT RULES
--------------------------------

✔ Colors MUST reflect meaning  
✔ Color order MUST match data order  
✔ Same meaning MUST always use same color  
✔ No random color assignment  

--------------------------------
FORBIDDEN
--------------------------------

✘ Mixing brand + sentiment colors without logic  
✘ Using green/red when no sentiment exists  
✘ Random gradients without scale meaning  

========================
NUMERIC HANDLING
========================
Numeric conversion rules apply ONLY when showData = true 
If input provides:

distribution → use
options → map
percentages → normalize
counts → convert to %

If partial:
→ Conservative estimation
→ Normalize to 100
→ Reasonable rounding

Never output missing values.

---
========================
PARTICIPATION
========================

The Participation & Demographics section MUST ALWAYS be generated.

Participation metrics represent survey logistics, not individual responses.

Therefore participation data is NOT subject to masking rules.

Always display the REAL values for:

invited
responded
rate

even when total_answers < 7.

These values reflect survey participation, not response content,
and do not expose individual answers.

Never omit or anonymize participation data.

If participation information is missing from input,
estimate realistic participation numbers using:

total_answers
survey size
typical response rates.

========================
ENTERPRISE KPI DETECTION ENGINE
========================

The dashboard MUST generate a single leadership KPI
in the "rightBlock" section.

The KPI represents the strongest available signal
of respondent advocacy or sentiment.

The AI must dynamically determine which KPI
can be calculated from the survey analysis.

The KPI must follow the decision hierarchy below.

KPI PRIVACY RULE (STRICT)

KPI behavior MUST follow GLOBAL showData decision.

IF GLOBAL showData = false:

→ KPI MUST be qualitative only  
→ NO numeric values  

IF GLOBAL showData = true:

→ KPI can include numeric values and chart  

--------------------------------
IMPORTANT
--------------------------------

KPI MUST NOT override privacy decisions  
Privacy ALWAYS takes priority over KPI visibility

--------------------------------
STEP 1 — NET PROMOTER SCORE
--------------------------------

If the survey contains a question measuring
likelihood to recommend the subject of the survey.

The subject must be inferred dynamically
from the survey_name, survey_overview,
or question wording.

using a 0–10 scale:

Example question patterns:

Example question patterns:

• How likely are you to recommend
• Likelihood to recommend
• Would you recommend
• Recommend this product
• Recommend this service
• Recommend this company
• Recommend this organization
• Recommend ...

Then calculate Net Promoter Score.

Classification:

Promoters → scores 9–10  
Passives → scores 7–8  
Detractors → scores 0–6  

Formula:

NPS = %Promoters − %Detractors

If NPS is available:

rightBlock.title = "Net Promoter Score"

The donut chart MUST display:

Promoters  
Passives  
Detractors

STRICT VALIDATION:

Net Promoter Score can ONLY be generated IF:

✔ A question explicitly measures recommendation likelihood
✔ AND uses a 0–10 scale

IF these conditions are NOT met:

→ Net Promoter Score is STRICTLY FORBIDDEN  
→ DO NOT attempt calculation  

→ MUST fallback to Service Advocacy Score

--------------------------------
STEP 2 — SERVICE ADVOCACY SCORE
--------------------------------
If the survey does NOT contain an NPS question but
If the survey contains quantitative evaluation signals,
the AI MUST generate a Service Advocacy Score.

COMPUTATION ENGINE (MANDATORY):

→ Identify ALL rating scale and satisfaction-type questions

→ For each question:
   • Normalize distribution into sentiment buckets:
     Positive = top scale values
     Neutral = midpoint values
     Negative = lowest scale values

→ Bucket boundaries MUST be derived dynamically:
   • top 40% of scale → Positive
   • middle range → Neutral
   • bottom 40% → Negative

→ Aggregate counts across ALL questions

→ If multiple questions exist:
   • Weight equally unless strong dominance detected
   • Dominance = one question contributes >60% of total responses

→ Final KPI distribution:
   series = [Positive, Neutral, Negative]

→ chartType = "donut"

→ options.labels = ["Positive", "Neutral", "Negative"]

→ options.colors MUST follow sentiment color rules

STRICT:

If showData = true:
→ chart MUST be generated using aggregated distribution
→ chart MUST NOT be empty

No static mappings allowed.
No single-question shortcut allowed.

--------------------------------
STEP 3 — SENTIMENT INDEX
--------------------------------

If the survey contains only qualitative feedback
or lacks rating signals, generate a Sentiment Index.

The Sentiment Index represents the emotional tone
of respondent feedback.

Sentiment groups:

Positive  
Neutral  
Negative

Sentiment signals may be extracted from:

• qualitative summaries
• feedback percentages
• positive vs negative language

If no explicit percentages exist,
infer sentiment direction conservatively.

rightBlock.title = "Survey Sentiment Index"

The donut chart MUST visualize:

Positive  
Neutral  
Negative

--------------------------------
RIGHT BLOCK NOTE (DYNAMIC GENERATION)
--------------------------------

The field rightBlock.note MUST be dynamically generated by the AI.

The note must briefly explain:

• what the KPI represents
• how the KPI was derived from the survey responses
• what the donut chart visualizes

The explanation must be written as a concise executive description
of the visualization.

Rules:

• Exactly 1 sentence
• 15–30 words
• Must describe the data source used for the KPI
• Must explain the sentiment groups visualized in the donut chart
• Must be generated dynamically based on the survey analysis
• MUST NOT copy template or example phrases

The note should follow this structure:

Sentence Structure Guidance:

Part 1 → Identify the KPI source using the survey context
(example: recommendation ratings for the product, satisfaction ratings for the service, qualitative feedback about the program)

Part 2 → Explain what the chart visualizes  
(example: distribution of promoters/passives/detractors or positive/neutral/negative sentiment)

The wording must be generated uniquely for each survey
based on the KPI selected by the ENTERPRISE KPI DETECTION ENGINE.
========================
RIGHT BLOCK
========================
RIGHT BLOCK VISIBILITY RULE (STRICT + ENTERPRISE SAFE)

The rightBlock MUST ALWAYS include:

✔ "showData"
✔ "title"
✔ "note"
✔ "chart"

--------------------------------
WHEN showData = true
--------------------------------

→ KPI MUST be calculated
CHART GENERATION (HARD REQUIREMENT)

IF showData = true:

→ chart MUST be present  
→ chart MUST contain:
   • chartType
   • series
   • options  

IF chart = {} OR missing ANY field:

→ OUTPUT INVALID  
→ MUST REGENERATE  

NO EXCEPTIONS
→ numeric values allowed

WHEN showData = false (STRICT ENTERPRISE MODE)
--------------------------------

→ KPI MUST still be IDENTIFIED (title REQUIRED)

→ KPI MUST be described QUALITATIVELY ONLY  
→ NO numeric values allowed  

--------------------------------
CHART ENFORCEMENT (STRICT)
--------------------------------

→ chart MUST be EXACTLY:
   "chart": {}

✘ chart MUST NOT contain:
   • chartType  
   • series  
   • options  
   • labels  
   • colors  

✘ ANY chart data presence = INVALID  

--------------------------------
VALIDATION
--------------------------------

If chart contains ANY data when showData = false:

→ Output is INVALID

--------------------------------
TITLE RULE (MANDATORY)
--------------------------------

title MUST NEVER be empty.

It MUST be one of:

• "Net Promoter Score"
• "Service Advocacy Score"
• "Survey Sentiment Index"

If empty → INVALID

--------------------------------
NOTE RULE (STRICT)
--------------------------------

The note MUST:

✔ Be 1 sentence (15–30 words)
✔ Mention KPI source (ratings / feedback / sentiment)
✔ Describe what would be visualized (even if chart hidden)

--------------------------------
VALIDATION
--------------------------------

If:
• title is empty
• chart contains data when showData = false
• numeric KPI exposed when showData = false

→ Output is INVALID
========================
FINAL VALIDATION
========================

Before returning:

✔ No nulls  
✔ No empty arrays   
✔ Unknown semantic meaning MUST be preserved, but wording MUST still follow meta.language translation rules 
✔ JSON parses  

✔ Every question MUST contain:
   - hasText
   - hasChart
   - chart
   - chartNote

✔ hasText MUST be boolean  
✔ hasChart MUST be boolean  

IF ANY of these fields are missing:
→ Output is INVALID  
→ MUST be regenerated

✔ If total_answers ≥ 3 AND insights contains text → hasText must be true
IF total_answers < 3 AND hasText = true:
→ INVALID
→ MUST be corrected to false BEFORE output (NO EXCEPTIONS)
→ MUST be corrected to false BEFORE output
✔ If insights is empty → hasText must be false

If any rule fails → regenerate.
--------------------------------
QUESTION STRUCTURE VALIDATION (CRITICAL)
--------------------------------

FOR EACH question:

✔ "insights" field MUST exist  
✔ insights MUST be an array  
✔ insights MUST NOT be missing  

IF missing:
→ OUTPUT INVALID  
→ MUST regenerate  

GLOBAL SHOWDATA VALIDATION (STRICT - FINAL AUTHORITY)

GLOBAL showData MUST be applied EXACTLY as follows:

IF total_respondents ≤ 3:

→ summary.showData = false  
→ rightBlock.showData = false  
→ participation.showData = false  
→ ALL questions.showData = false  
→ ALL charts MUST be {}

ELSE:

→ summary.showData = true  
→ rightBlock.showData = true  
→ participation.showData = true  
→ ALL questions.showData = true  

→ Charts MUST be generated normally
→ hasChart MUST follow chart presence

========================
OUTPUT
========================

Return ONLY the final JSON.
No explanation.
No markdown.

Begin transformation.   
"""



# ── Merge helper (mirrors JS logic) ──────────────────────────────────────────

import re as _re

def _extract_json(raw: str) -> str:
    """Strip markdown fences and extract first JSON object from a string."""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    match = _re.search(r'\{[\s\S]*\}', cleaned)
    return match.group(0) if match else cleaned


def _has_error(raw) -> bool:
    if not raw:
        return True
    try:
        text = raw if isinstance(raw, str) else json.dumps(raw)
        data = json.loads(_extract_json(text))
        return "error" in data
    except Exception:
        return True


def merge_report_outputs(
    report_out,   # dict or raw JSON string from Report Writer agent
    risk_out,     # dict or raw JSON string from Risk agent
) -> tuple[dict, str]:
    """
    Merge report and risk outputs exactly like the JS pipeline:
      merged = { ...reportJSON, ...riskJSON }
      record_id = merged?.meta?.record_id ?? ""

    If either output contains an error key, returns the report output unchanged.
    Returns (merged_dict, record_id).
    """
    def to_dict(v):
        if isinstance(v, dict):
            return v
        return json.loads(_extract_json(v))

    if _has_error(report_out) or _has_error(risk_out):
        safe = to_dict(report_out) if not _has_error(report_out) else {}
        return safe, safe.get("meta", {}).get("record_id", "")

    report_json = to_dict(report_out)
    risk_json   = to_dict(risk_out)

    merged    = {**report_json, **risk_json}          # JS: {...reportJSON, ...topRiskJSON}
    record_id = merged.get("meta", {}).get("record_id", "")
    return merged, record_id


class AnalysisService:
    def __init__(self, settings: OpenAISettings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.api_key)

    # ── Per-cluster analysis ──────────────────────────────────────────────────

    async def analyze_cluster(
        self,
        cluster: ClusterAnalysis,
        question_text: str,
        *,
        is_noise: bool = False,
    ) -> ClusterAnalysis:
        """
        Analyze one cluster (or the noise group) and return an enriched copy.
        Uses gpt-4o-mini — high volume, cost-sensitive.
        """
        responses_text = "\n".join(f"- {q}" for q in cluster.representative_quotes)

        cluster_type = (
            "Diverse / unclassified responses (did not form a tight cluster)"
            if is_noise else
            "Clustered responses (share a common idea)"
        )
        user_msg = CLUSTER_USER.format(
            question_text=question_text,
            cluster_type=cluster_type,
            responses=responses_text,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.analysis_model,
                messages=[
                    {"role": "system", "content": CLUSTER_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception as exc:
            raise AnalysisError(f"Cluster analysis LLM call failed: {exc}") from exc

        try:
            return cluster.model_copy(update={"theme": data.get("theme", "")})
        except Exception as exc:
            raise AnalysisError(
                f"Cluster analysis response parsing failed: {exc}. Raw: {raw}"
            ) from exc

    async def analyze_all_clusters(
        self,
        question_result: QuestionClusterResult,
        question_text: str,
        *,
        include_noise: bool = True,
        max_concurrent: int = 5,
    ) -> QuestionClusterResult:
        """
        Analyze every cluster for a question concurrently.
        If include_noise=True and there are noise responses, wraps them as a
        synthetic cluster so no responses are silently dropped.
        """
        clusters = list(question_result.clusters)

        # Synthesize a noise cluster from unclassified responses
        noise_cluster: ClusterAnalysis | None = None
        if include_noise and question_result.noise_count > 0:
            noise_quotes = [
                lbl_resp
                for lbl_resp, lbl in zip(
                    [r.response_text or "" for r in self._get_noise_responses(question_result)],
                    question_result.cluster_labels,
                )
                if lbl.cluster_id == -1 and lbl_resp
            ]
            if noise_quotes:
                noise_cluster = ClusterAnalysis(
                    cluster_id=-1,
                    representative_quotes=noise_quotes[:10],  # cap at 10 for prompt size
                    response_count=question_result.noise_count,
                    percentage_of_total=round(
                        question_result.noise_count / question_result.total_responses * 100, 2
                    ),
                )

        semaphore = asyncio.Semaphore(max_concurrent)
        analyzed: list[ClusterAnalysis] = [None] * len(clusters)  # type: ignore

        async def process(idx: int, cluster: ClusterAnalysis) -> None:
            async with semaphore:
                analyzed[idx] = await self.analyze_cluster(cluster, question_text)

        await asyncio.gather(*[process(i, c) for i, c in enumerate(clusters)])

        # Analyze noise cluster separately (different system prompt)
        analyzed_noise: ClusterAnalysis | None = None
        if noise_cluster:
            async with semaphore:
                analyzed_noise = await self.analyze_cluster(
                    noise_cluster, question_text, is_noise=True
                )

        final_clusters = analyzed
        if analyzed_noise:
            final_clusters = analyzed + [analyzed_noise]

        return question_result.model_copy(update={"clusters": final_clusters})

    def _get_noise_responses(self, result: QuestionClusterResult):
        """Placeholder — real implementation uses stored response texts."""
        return []

    # ── Data analyst agent ────────────────────────────────────────────────────

    async def analyze_question(
        self,
        question_id: str,
        question_text: str,
        all_responses: list[str],
        numeric_context: dict | None = None,
    ) -> dict:
        """
        Deep analysis of one question. Uses gpt-4o.

        numeric_context (optional) — pass for numeric questions:
            {
              "question_type": "single choice" | "multiple choice" | ...,
              "options": { "Option A": 5, "Option B": 3, ... },
              "total_answers": 14,
            }
        When provided, the analyst sees both the aggregated option counts AND the
        open-ended explanations, and is instructed to connect them.
        """
        if numeric_context:
            options = numeric_context.get("options", {})
            total   = numeric_context.get("total_answers", len(all_responses))
            qtype   = numeric_context.get("question_type", "structured")

            # Format options as a readable block with percentages.
            # Some question types (ranking, rating) may have nested or zero values.
            options_lines = []
            for opt, count in sorted(
                options.items(),
                key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0,
            ):
                if isinstance(count, (int, float)):
                    pct = round(count / total * 100, 1) if total else 0
                    options_lines.append(f"  {opt}: {count} respondents ({pct}%)")
                else:
                    options_lines.append(f"  {opt}: {json.dumps(count)}")
            options_block = "\n".join(options_lines) or "  (no aggregated option data)"

            explanations = [r for r in all_responses if r.strip()]
            if explanations:
                exp_block = "\n".join(f"{i+1}. {r}" for i, r in enumerate(explanations))
            else:
                exp_block = "  (no written explanations provided)"

            user_msg = DATA_ANALYST_USER_NUMERIC.format(
                question_text=question_text,
                question_type=qtype,
                total=total,
                options_block=options_block,
                exp_count=len(explanations),
                explanations_block=exp_block,
            )
        else:
            numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(all_responses))
            user_msg = DATA_ANALYST_USER_OPEN.format(
                question_text=question_text,
                total=len(all_responses),
                responses=numbered,
            )

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.completion_model,
                messages=[
                    {"role": "system", "content": DATA_ANALYST_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            data["question_id"]     = question_id
            data["question_text"]   = question_text
            data["total_responses"] = numeric_context.get("total_answers", len(all_responses)) if numeric_context else len(all_responses)
            data["question_type"]   = numeric_context.get("question_type", "open_ended") if numeric_context else "open_ended"
            logger.info("question_analysis_done", question_id=question_id)
            return data
        except Exception as exc:
            raise AnalysisError(f"Question analysis failed for {question_id}: {exc}") from exc

    async def analyze_all_questions(
        self,
        questions: dict[str, tuple[str, list[str], dict | None]],
        max_concurrent: int = 10,
    ) -> list[dict]:
        """
        Analyses every question in parallel — one GPT call per question.
        Scales to any number of questions regardless of response count.
        questions = { qid: (question_text, [responses], numeric_context | None) }
        Returns list of dicts sorted ascending by question_id.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _analyse_one(qid: str) -> dict:
            qtext, resps, num_ctx = questions[qid]
            async with semaphore:
                result = await self.analyze_question(qid, qtext, resps, num_ctx)
            summary = result if "summary" not in result else result
            summary.setdefault("question_id",     qid)
            summary.setdefault("question_text",   qtext)
            summary.setdefault("question_type",   num_ctx.get("question_type", "open_ended") if num_ctx else "open_ended")
            summary.setdefault("total_responses", num_ctx.get("total_answers", len(resps)) if num_ctx else len(resps))
            logger.info("question_analysis_done", question_id=qid)
            return {"question_id": qid, "summary": summary}

        sorted_qids = sorted(questions.keys())
        results = await asyncio.gather(*[_analyse_one(qid) for qid in sorted_qids])
        return list(results)

    # ── Report-level generation (gpt-4o) ─────────────────────────────────────

    async def generate_executive_summary(self, as_json_report: str) -> str:
        user_msg = EXECUTIVE_USER.format(as_json_report=as_json_report)

        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("AI INPUT ▶ EXECUTIVE SUMMARY AGENT")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("executive_summary_model", model=self._settings.completion_model)
        # logger.info("executive_summary_system_prompt", system_prompt=EXECUTIVE_SYSTEM)
        logger.info("executive_summary_user_message", user_message=user_msg)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.completion_model,
                messages=[
                    {"role": "system", "content": EXECUTIVE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.4,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise AnalysisError(f"Executive summary generation failed: {exc}") from exc

    async def generate_report(
        self,
        analyses: list[dict],
        meta: dict,
        raw_data: dict | None = None,
        recommendations: dict | None = None,
    ) -> dict:
        combined = self._build_combined_input(
            analyses, meta, raw_data or {}, recommendations or {}
        )
        combined_json = json.dumps(combined, indent=2, ensure_ascii=False)

        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("AI INPUT ▶ REPORT WRITER AGENT")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("report_writer_model", model=self._settings.completion_model)
        logger.info("report_writer_questions",
                    count=len(combined[0].get("question_analysis", {})),
                    question_ids=list(combined[0].get("question_analysis", {}).keys()))
        logger.info("report_writer_full_input", input=combined_json)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        user_msg = combined_json
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.completion_model,
                messages=[
                    {"role": "system", "content": REPORT_WRITER_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            result = json.loads(response.choices[0].message.content or "{}")
            logger.info("report_writer_done",
                        output_keys=list(result.keys()),
                        questions_count=len(result.get("questions", [])))
            return result
        except Exception as exc:
            raise AnalysisError(f"Report generation failed: {exc}") from exc

    @staticmethod
    def _build_combined_input(
        analyses: list[dict],
        meta: dict,
        raw_data: dict,
        recommendations: dict,
    ) -> list[dict]:
        """Assemble the exact JSON format the report writer agent expects."""
        # Index analysis results by question_id
        by_id: dict[str, dict] = {}
        for a in analyses:
            qid = a.get("question_id", "")
            by_id[qid] = a.get("summary", a)

        question_analysis: dict[str, dict] = {}

        # Numeric questions (single choice, multiple choice, rating, ranking)
        for qid, q in raw_data.get("numeric_questions", {}).items():
            s = by_id.get(qid, {})
            entry: dict = {
                "question":       q.get("question_text", ""),
                "question_topic": q.get("question_topic", ""),
                "type":           q.get("question_type", "structured"),
                "total_answers":  q.get("total_answers", 0),
            }
            if q.get("distribution"):
                entry["scale"]        = q.get("scale", {"min": 1, "max": 5})
                entry["distribution"] = q["distribution"]
            else:
                entry["options"] = AnalysisService._flatten_options(q.get("options", {}))

            findings = s.get("key_findings") or s.get("response_patterns") or []
            if isinstance(findings, str):
                findings = [findings]
            entry["summary"] = findings[:6]
            question_analysis[qid] = entry

        # Open-ended questions
        for q in raw_data.get("text_questions", []):
            qid   = q["question_id"]
            s     = by_id.get(qid, {})
            n_ans = sum(1 for r in q.get("responses", []) if r.get("answer", "").strip())
            findings = s.get("key_findings") or s.get("response_patterns") or []
            if isinstance(findings, str):
                findings = [findings]
            question_analysis[qid] = {
                "question":       q.get("question_text", ""),
                "question_topic": q.get("question_topic", ""),
                "type":           "open",
                "total_answers":  n_ans,
                "summary":        findings[:6],
            }

        # Sort by question ID so order is Q1, Q2 ... Q10
        question_analysis = dict(sorted(question_analysis.items()))

        return [{
            "meta":              meta,
            "question_analysis": question_analysis,
            "recommendations":   recommendations,
        }]

    @staticmethod
    def _flatten_options(options: dict) -> dict:
        """
        Normalize FileMaker option counts.
        FileMaker sometimes splits a key at special characters, producing:
          "Facilities (cafes, restrooms, etc": { ")": 0 }
        This reconstructs the full key and ensures every value is an int.
        """
        result = {}
        for key, value in options.items():
            if isinstance(value, dict):
                for suffix, count in value.items():
                    full_key = key + suffix
                    result[full_key] = int(count) if isinstance(count, (int, float)) else 0
            else:
                result[key] = int(value) if isinstance(value, (int, float)) else 0
        return result

    @staticmethod
    def _infer_total_respondents(analyses: list[dict], meta: dict) -> int:
        if meta.get("survey_recipients"):
            try:
                return int(meta["survey_recipients"])
            except (ValueError, TypeError):
                pass
        for a in analyses:
            s = a.get("summary", a)
            tr = s.get("total_responses") or s.get("total_answers")
            if tr:
                try:
                    return int(tr)
                except (ValueError, TypeError):
                    pass
        return 0

    async def generate_risk_and_action_plan(
        self,
        analyses: list[dict],
        raw_data: dict | None = None,
        meta: dict | None = None,
    ) -> dict:
        """
        Single Risk agent call — returns one JSON containing both
        risk identification and action plan.
        Input: meta + question_analysis (same structure as report writer, no recommendations).
        """
        combined = self._build_combined_input(
            analyses, meta or {}, raw_data or {}, {}
        )
        # Risk agent receives meta + question_analysis only — no recommendations key
        risk_input = {"meta": combined[0]["meta"], "question_analysis": combined[0]["question_analysis"]}
        combined_json = json.dumps([risk_input], indent=2, ensure_ascii=False)

        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("AI INPUT ▶ RISK + ACTION PLAN AGENT")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("risk_agent_model", model=self._settings.completion_model)
        logger.info("risk_agent_full_input", input=combined_json)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        user_msg = RISK_USER.format(combined_input=combined_json)
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.completion_model,
                messages=[
                    {"role": "system", "content": RISK_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            return json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:
            raise AnalysisError(f"Risk + action plan generation failed: {exc}") from exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _s(a: dict) -> dict:
        """Return the summary sub-dict from an analysis entry."""
        return a.get("summary", a)

    def _build_themes_from_analyses(self, analyses: list[dict]) -> str:
        sorted_analyses = sorted(analyses, key=lambda x: x.get("question_id", ""))
        lines = []
        for a in sorted_analyses:
            s = self._s(a)
            lines.append(f"[{a['question_id']}] {s.get('question_text', '')}")
            lines.append(f"  Sentiment     : {s.get('sentiment', '—')}")
            lines.append(f"  Overview      : {s.get('overview', s.get('summary','—'))}")
            lines.append(f"  Majority view : {s.get('majority_view', '—')}")
            for f in s.get("key_findings", [])[:3]:
                lines.append(f"  Finding       : {f}")
            lines.append("")
        return "\n".join(lines)

    def _build_structured_from_analyses(self, analyses: list[dict]) -> str:
        structured = [a for a in analyses if self._s(a).get("question_type") != "open_ended"]
        lines = []
        for a in sorted(structured, key=lambda x: x.get("question_id", "")):
            s = self._s(a)
            lines.append(
                f"[{a['question_id']}] {s.get('question_text','')} ({s.get('question_type','—')}): "
                f"{s.get('majority_view', '—')}"
            )
        return "\n".join(lines)
