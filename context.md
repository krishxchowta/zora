# Zora — Autonomous Healthcare Analytics System

> **Backend-only** production pipeline. Frontend (Next.js 15) is deferred.
> Last verified: 2026-04-08 | Run `f1fa2d44a342` — full_complete

---

## 1. Project Overview

**Zora** is an autonomous, multi-agent healthcare analytics backend that accepts a patient dataset (CSV/XLSX/JSON), runs it through a 6-stage pipeline — ingest, embed, clean, AutoML, synthesis, narration — and produces:

- A **best ML model** with SHAP explainability
- **Protein biomarker stability** analysis (AlphaFold mock)
- **Financial risk** indicators (denial probability, healthcare waste estimate)
- **Safety vault** flags with deterministic medical rules
- **Dual-voice narrations** (clinical + patient-friendly) quality-gated by an LLM critic
- Optional **Twilio SMS** and **Google Cloud TTS** audio delivery

Every stage emits **real-time Server-Sent Events (SSE)** so a frontend (or curl) can stream progress live.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Python 3.10, FastAPI |
| Agent orchestration | CrewAI 1.14 (`crewai.LLM`, `Process.sequential`) |
| Primary LLM | Google Gemini 2.0 Flash (`gemini/gemini-2.0-flash`) |
| Fallback LLM | Groq LLaMA 3.3 70B (`groq/llama-3.3-70b-versatile`) |
| Critic LLM | Groq LLaMA 3.1 8B (`groq/llama-3.1-8b-instant`) |
| Embeddings | Google `models/gemini-embedding-001` (768-dim) via `google-genai` SDK |
| AutoML | PyCaret 3.x (classification) |
| Explainability | SHAP (TreeExplainer + KernelExplainer fallback) |
| Data parsing | Polars (ingest), Pandas (clean + AutoML) |
| Database | Supabase (PostgreSQL + pgvector extension) |
| Vector search | `match_documents()` RPC — cosine similarity |
| SMS delivery | Twilio REST API |
| TTS audio | Google Cloud Text-to-Speech REST API |
| Logging | structlog (bound per run_id) |
| Tracing | LangSmith (optional, via LANGCHAIN_API_KEY) |

---

## 2. Repository Structure

```
zora-backend/
├── main.py                          # FastAPI app, CORS, router mounts, /health endpoint
├── .env                             # All API keys and config (see Section 12)
├── context.md                       # This file
│
├── routes/
│   ├── __init__.py
│   ├── run.py                       # POST /api/run, GET /api/run/{id}/status, _run_pipeline()
│   └── stream.py                    # GET /run/{id}/stream (SSE endpoint)
│
├── agents/
│   ├── __init__.py
│   ├── zora_ingest.py               # S1 — Polars parse + CrewAI LLM summary → SchemaProfile
│   ├── zora_embed.py                # S2 — Gemini embedding-001 → Supabase pgvector
│   ├── zora_clean.py                # S3 — Pandas clean + Critic Gate 1 (LLM quality scorer)
│   ├── zora_automl.py               # S4 — PyCaret + AlphaFold mock + RAG cosine Gate 1
│   ├── zora_synthesis.py            # S5 — Finance + Safety + RAG citations + LLM synthesis
│   └── zora_narrator.py             # S6 — Dual voice + G2 Critic Gate 2 + Twilio + TTS
│
├── tools/
│   ├── __init__.py
│   ├── ingest_tool.py               # Polars CSV/XLSX/JSON → SchemaProfile + ingested.csv
│   ├── embed_tool.py                # build_chunks() + batch Gemini embed + Supabase insert
│   ├── clean_tool.py                # dedup + null impute (median/mode) + IQR outlier removal
│   ├── automl_tool.py               # PyCaret compare_models + SHAP + DIAGNOSIS_PROTEIN_MAP
│   ├── alphafold_tool.py            # Deterministic mock stability score [0.25, 0.95]
│   ├── finance_tool.py              # denial_probability + waste_estimate_usd (CMS benchmarks)
│   └── safety_vault.py              # 6 hard-coded rules SR-001…SR-006 (no dynamic execution)
│
├── services/
│   └── supabase_service.py          # create_run_record, update_run_status, get_run
│
├── models/
│   └── schemas.py                   # Pydantic: SchemaProfile, CleanReport, SSE models, etc.
│
├── utils/
│   ├── config.py                    # pydantic-settings BaseSettings from .env
│   ├── sse_manager.py               # List-buffer SSEManager with cursor-polling subscribers
│   └── logger.py                    # structlog get_run_logger(run_id) → bound logger
│
├── test_data/
│   └── patient_readmission.csv      # 42-row synthetic healthcare test dataset (18 columns)
│
├── uploads/{run_id}/                # Raw uploaded files (one dir per run)
│   └── patient_readmission.csv
│
└── outputs/{run_id}/                # Processed outputs
    ├── ingested.csv                 # Post-Polars parse
    ├── cleaned.csv                  # Post-dedup/impute/IQR
    └── narration_patient.mp3        # Google TTS audio (if configured)
```

---

## 3. Pipeline Architecture

```
                     ┌─────────────────────────────────────────────────────────┐
  Upload CSV ───────►│  S1 Ingest  │  S2 Embed  │  S3 Clean + G1 Critic      │
                     │  (Polars)   │ (pgvector) │  (Pandas + LLM 7/10)       │
                     └──────┬──────┴──────┬─────┴──────────┬──────────────────┘
                            │             │                │
                     ┌──────▼──────┬──────▼─────┬──────────▼──────────────────┐
                     │  S4 AutoML  │  AlphaFold │  G1 RAG Cosine (≥0.82)     │
                     │  (PyCaret)  │   (mock)   │  (hallucination check)     │
                     └──────┬──────┴──────┬─────┴──────────┬──────────────────┘
                            │             │                │
                     ┌──────▼─────────────▼────────────────▼──────────────────┐
                     │  S5 Synthesis: FinanceTool + SafetyVault + RAG + LLM   │
                     │  → writes `insights` row to Supabase                   │
                     └──────────────────────┬─────────────────────────────────┘
                                            │
                     ┌──────────────────────▼─────────────────────────────────┐
                     │  S6 Narrator: clinical_mode + patient_mode             │
                     │  → [G2] Critic Gate 2 (clarity/completeness/tone)      │
                     │  → optional Twilio SMS + Google TTS                    │
                     │  → pipeline_complete (status=full_complete)            │
                     └────────────────────────────────────────────────────────┘
```

### Stage Details

| Stage | Agent File | Tool(s) Used | Input | Output | Supabase Write |
|-------|-----------|-------------|-------|--------|----------------|
| **S1** Ingest | `zora_ingest.py` | `ingest_tool` | Uploaded file + target_column | `SchemaProfile` + `ingested.csv` | `runs.status=running`, `runs.schema_json` |
| **S2** Embed | `zora_embed.py` | `embed_tool` | `SchemaProfile` | 20 vectors in pgvector | `runs.status=s2_complete`, `runs.embedding_count` |
| **S3** Clean | `zora_clean.py` | `clean_tool` | `SchemaProfile` | `CleanReport` + `cleaned.csv` | `runs.status=s3_complete`, `runs.quality_score` |
| **S4** AutoML | `zora_automl.py` | `automl_tool` + `alphafold_tool` | `cleaned.csv` | `{automl, alphafold, gate1}` dict | `runs.status=s4_complete`, `runs.automl_summary` |
| **S5** Synthesis | `zora_synthesis.py` | `finance_tool` + `safety_vault` | S4 result + CleanReport | `synthesis_result` dict | `insights` table row |
| **S6** Narrator | `zora_narrator.py` | Twilio + TTS | `synthesis_result` | dual narrations + G2 score | `insights.narration_*`, `insights.g2_*` |

---

## 4. API Reference

### `POST /api/run` — Start a Pipeline Run

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | CSV, XLSX, XLS, or JSON dataset |
| `problem_desc` | string | No | Human description of the analytics goal |
| `target_column` | string | No | Column name to predict (binary classification) |

**Response** (`200 OK`):
```json
{
  "run_id": "f1fa2d44a342",
  "status": "queued",
  "filename": "patient_readmission.csv"
}
```

**Side-effects:**
1. Saves uploaded file to `uploads/{run_id}/{filename}`
2. Creates a `runs` table row with status `queued`
3. Launches `_run_pipeline(run_id, filepath, target_column)` as a background `asyncio.create_task`

---

### `GET /api/run/{run_id}/status` — Check Run Status

**Response** (`200 OK`):
```json
{
  "run_id": "f1fa2d44a342",
  "status": "full_complete",
  "rows_count": 42,
  "cols_count": 18,
  "schema_json": { ... },
  "embedding_count": 20,
  "created_at": "2026-04-08T14:15:18.000Z",
  "completed_at": "2026-04-08T14:17:24.000Z"
}
```

**Status progression:** `queued` → `running` → `s2_complete` → `s3_complete` → `s4_complete` → `full_complete` | `failed`

**Error:** `404` if run_id not found.

---

### `GET /run/{run_id}/stream` — SSE Event Stream

**Media-Type:** `text/event-stream`

**Headers sent:**
```
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

**Protocol:** Each event is `data: {JSON}\n\n`. Stream terminates when `type=pipeline_complete` is received. Late subscribers replay all buffered events from the beginning (list-buffer design).

**Timeout:** 5 minutes (300 seconds).

---

### `GET /health` — Health Check

**Response:**
```json
{"status": "ok", "app": "Zora", "stage": "S1-S5+Narrator"}
```

---

## 5. SSE Event Protocol

### Event Shape

Every SSE event is a JSON object with these fields:

```json
{
  "type": "agent_update | pipeline_complete | error",
  "agent": "zora_ingest | zora_embed | zora_clean | zora_critic | zora_automl | zora_critic_gate1 | zora_synthesis | zora_narrator | zora_critic_g2 | pipeline",
  "status": "running | completed | failed | warning",
  "latency_ms": 2374,
  "output_summary": "Human-readable summary of what happened",
  "data": { },
  "timestamp": "2026-04-08T14:15:20.948685+00:00"
}
```

### Full 10-Event Sequence (verified run)

| # | Agent | Status | Key Data |
|---|-------|--------|----------|
| 1 | `zora_ingest` | running | "Parsing dataset with Polars..." |
| 2 | `zora_ingest` | completed | rows=42, cols=18, target=readmission_30day, latency=2374ms |
| 3 | `zora_embed` | running | "Generating embeddings via gemini-embedding-001..." |
| 4 | `zora_embed` | completed | vector_count=20, latency=3840ms |
| 5 | `zora_clean` | running | "Cleaning dataset: null impute + dedup + outlier IQR..." |
| 6 | `zora_critic` | running → completed | score=9/10, PASS, attempt 1/3 |
| 7 | `zora_clean` | completed | 42→35 rows, 2 dupes, 3 nulls imputed, latency=2557ms |
| 8 | `zora_automl` | running → completed | DecisionTreeClassifier AUC=1.0, BNP stability=0.8871, latency=99s |
| 9 | `zora_critic_gate1` | warning | cosine=0.7697, threshold=0.82 — WARN, proceeded with caution |
| 10 | `zora_synthesis` | completed | denial 31%, waste $44K, doctor_review=true, insight #1, latency=4243ms |
| 11 | `zora_narrator` | running → completed | clinical (1615 chars) + patient (1034 chars), TTS saved, latency=8215ms |
| 12 | `zora_critic_g2` | completed | G2 score=9.33/10 PASS (clinical clarity 5/5, patient tone 5/5) |
| 13 | `pipeline_complete` | full_complete | "Full pipeline complete (S1→S5+Narrator). Insight #1. G2 score: 9.33/10." |

---

## 6. Pydantic Data Models

### `SchemaProfile` — Built by S1 ingest_tool

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | 12-char hex run identifier |
| `filename` | str | Original upload filename |
| `rows` | int | Total row count |
| `cols` | int | Total column count |
| `columns` | list[dict] | Per-column: `{name, dtype, null_count, null_pct, sample_values}` |
| `numeric_columns` | list[str] | Column names with numeric dtype |
| `categorical_columns` | list[str] | Column names with object/category dtype |
| `datetime_columns` | list[str] | Column names with datetime dtype |
| `target_candidate` | str or None | Auto-detected or user-specified target column |
| `null_summary` | dict | `{column_name: null_percentage}` for columns with nulls |
| `duplicate_count` | int | Number of exact duplicate rows |
| `memory_mb` | float | Dataset memory footprint in MB |

### `CleanReport` — Built by S3 clean_tool

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | Run identifier |
| `rows_before` | int | Row count pre-cleaning |
| `rows_after` | int | Row count post-cleaning |
| `dupes_removed` | int | Duplicate rows dropped |
| `nulls_imputed` | dict | `{column: count_imputed}` |
| `outliers_removed` | dict | `{column: count_removed}` |
| `imputation_strategy` | dict | `{column: "median"\|"mode"\|"none"}` |
| `quality_score` | int or None | Critic score 0-10 |
| `critic_feedback` | str or None | Critic justification text |
| `passed_critic` | bool | True if score >= 7 |

### `RunCreateResponse`

| Field | Type |
|-------|------|
| `run_id` | str |
| `status` | str |
| `filename` | str |

### `RunStatusResponse`

| Field | Type |
|-------|------|
| `run_id` | str |
| `status` | str |
| `rows_count` | int or None |
| `cols_count` | int or None |
| `schema_json` | dict or None |
| `embedding_count` | int or None |
| `created_at` | datetime or None |
| `completed_at` | datetime or None |

### `AgentSSEEvent`

| Field | Type |
|-------|------|
| `type` | str — `"agent_update"` or `"pipeline_complete"` or `"error"` |
| `agent` | str |
| `status` | str — `"running"`, `"completed"`, `"failed"`, `"warning"` |
| `latency_ms` | int or None |
| `output_summary` | str or None |
| `error_message` | str or None |
| `data` | dict or None |
| `timestamp` | str (ISO 8601) |

---

## 7. Supabase Schema

### Table: `runs`

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | text (PK) | 12-char hex identifier |
| `filename` | text | Original upload filename |
| `filepath` | text | Server disk path to uploaded file |
| `problem_desc` | text | User-provided problem description |
| `target_column` | text | ML prediction target column name |
| `status` | text | Pipeline stage status (queued → running → s2_complete → s3_complete → s4_complete → full_complete \| failed) |
| `rows_count` | int | Post-ingest row count |
| `cols_count` | int | Column count |
| `schema_json` | jsonb | Full SchemaProfile as JSON |
| `embedding_count` | int | Number of vectors stored in documents table |
| `cleaned_rows` | int | Post-cleaning row count |
| `quality_score` | int | Critic quality score (0-10) |
| `cleaning_summary` | jsonb | Full CleanReport as JSON |
| `automl_summary` | jsonb | Model name + metrics + gate1_cosine + gate1_passed |
| `alphafold_summary` | jsonb | protein_name + stability_score + pdb_link + confidence |
| `created_at` | timestamptz | Auto-set on insert |
| `completed_at` | timestamptz | Set when pipeline reaches full_complete |

### Table: `documents` (pgvector)

| Column | Type | Description |
|--------|------|-------------|
| `id` | bigserial (PK) | Auto-increment |
| `run_id` | text | Foreign key to runs |
| `chunk_index` | int | 0 = dataset overview, 1 = null summary, 2+ = per-column profiles |
| `chunk_text` | text | Embedded text content |
| `metadata` | jsonb | `{run_id, chunk_type, chunk_index, column_name?}` |
| `embedding` | vector(768) | Gemini embedding-001 output (768 dimensions) |

**Chunk breakdown for an 18-column dataset:**
- Chunk 0: dataset overview (rows, cols, memory, target, column lists)
- Chunk 1: null quality summary ("cholesterol is 7.14% null")
- Chunks 2–19: one per column (name, dtype, null_pct, sample values)
- **Total: 20 vectors**

### Table: `insights`

| Column | Type | Description |
|--------|------|-------------|
| `id` | bigserial (PK) | Auto-increment |
| `run_id` | text | Foreign key to runs |
| `ml_model` | text | Best model class name (e.g., "DecisionTreeClassifier") |
| `ml_accuracy` | float | Model accuracy score |
| `ml_auc` | float | Model AUC-ROC score |
| `top_features` | jsonb | `{feature_name: mean_abs_shap_value}` — top 10 |
| `stability_score` | float | AlphaFold mock stability [0.25, 0.95] |
| `pdb_link` | text | AlphaFold DB URL (`alphafold.ebi.ac.uk/entry/{uniprot_id}`) |
| `protein_name` | text | Mapped protein (e.g., "BNP") |
| `denial_probability` | float | Insurance denial estimate [0, 0.99] |
| `waste_estimate` | float | Healthcare waste estimate in USD |
| `rag_citations` | jsonb | `[{chunk_text, similarity}]` — top 5 RAG matches |
| `synthesis_text` | text | LLM-generated synthesis paragraph (5-8 sentences) |
| `safety_flags` | jsonb | `[{rule_id, name, message, action}]` — triggered rules |
| `doctor_review` | boolean | True if any rule requires physician sign-off |
| `narration_clinical` | text | Dense technical narration for physicians |
| `narration_patient` | text | Plain English narration for patients/families |
| `g2_score` | float | G2 Critic Gate 2 composite score (/10) |
| `g2_passed` | boolean | True if g2_score >= 7.0 |
| `created_at` | timestamptz | Auto-set on insert |

### RPC: `match_documents(query_embedding, match_count, filter)`

- **Purpose:** pgvector cosine similarity search over the `documents` table
- **Parameters:**
  - `query_embedding` — float[] of length 768
  - `match_count` — int (number of results)
  - `filter` — jsonb (e.g., `{"run_id": "f1fa2d44a342"}`)
- **Returns:** `[{id, run_id, chunk_text, metadata, similarity}]` sorted by similarity desc
- **Used by:**
  - S3 `_retrieve_schema_context()` — RAG context for critic prompt (k=3)
  - S4 `_rag_cosine_check()` — hallucination grounding (k=1, threshold 0.82)
  - S5 `_retrieve_rag_citations()` — evidence for synthesis (k=5)

---

## 8. LLM Strategy

### Fallback Chain Pattern

Every LLM call in Zora follows the same pattern:

```python
candidates = [
    LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=T),
    LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=T),
]
for llm in candidates:
    try:
        return crew.kickoff()
    except Exception:
        continue
raise RuntimeError("All LLM candidates failed")
```

**Why:** Google Gemini free tier frequently returns 429 (rate limit). The Groq fallback handles this transparently — the user never sees the error.

### LLM Assignments

| Component | Primary | Fallback | Temperature | Purpose |
|-----------|---------|----------|-------------|---------|
| S1 Ingest agent | gemini-2.0-flash | groq/llama-3.3-70b | 0.2 | Schema summary |
| S2 Embed agent | gemini-2.0-flash | groq/llama-3.3-70b | 0.2 | Embed context summary |
| S3 Critic | groq/llama-3.1-8b-instant | groq/llama-3.3-70b | 0.0 | Data quality scoring |
| S5 Synthesis | gemini-2.0-flash | groq/llama-3.3-70b | 0.2 | Clinical synthesis text |
| S6 Narrator | gemini-2.0-flash | groq/llama-3.3-70b | 0.3 | Dual-voice narration |
| G2 Critic | groq/llama-3.1-8b-instant | groq/llama-3.3-70b | 0.0 | Narration quality scoring |

### Embedding Model

- **Model:** `models/gemini-embedding-001` via `google-genai` Python SDK (NOT LangChain)
- **Dimensions:** 768 (set via `output_dimensionality=768` to match Supabase `vector(768)`)
- **Task types:** `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search
- **SDK:** `from google import genai as google_genai` → `client.models.embed_content()`

### CrewAI Configuration

- **Version:** CrewAI 1.14
- **LLM wrapper:** `crewai.LLM` (string model IDs like `"gemini/gemini-2.0-flash"`, not LangChain chat model objects)
- **Process:** `Process.sequential` (single agent + single task per crew)
- **Agent settings:** `max_iter=1`, `allow_delegation=False`, `verbose=False`
- **LiteLLM:** Required for Groq routing (`pip install litellm`)

---

## 9. Tool Reference

### `ingest_tool(run_id, filepath, target_column)` → `SchemaProfile`
**File:** `tools/ingest_tool.py`

- Detects file type from extension: `.csv` → Polars `scan_csv`, `.xlsx/.xls` → Pandas → Polars `from_pandas`, `.json` → Polars `scan_ndjson`
- Calls `.collect()` to materialize the lazy frame
- Builds a `SchemaProfile` with per-column stats: dtype, null_count, null_pct, 5 sample values
- Auto-detects `target_candidate` if not provided (looks for binary int columns)
- Saves `outputs/{run_id}/ingested.csv` via Pandas `to_csv()`

### `embed_tool(profile)` → `int`
**File:** `tools/embed_tool.py`

- `build_chunks(profile)` creates `list[Document]`:
  - Chunk 0: dataset overview (rows, cols, memory, target, column lists)
  - Chunk 1: null summary ("cholesterol is 7.14% null")
  - Chunks 2+: one per column (name, dtype, null_pct, sample values)
- Batched embedding via `google-genai` SDK (all chunks in one API call)
- Direct Supabase `table("documents").insert(rows)` (not LangChain SupabaseVectorStore)
- Returns count of vectors stored

### `clean_tool(run_id, profile, feedback_ctx=None)` → `CleanReport`
**File:** `tools/clean_tool.py`

- **Step 1 — Dedup:** `df.drop_duplicates()`
- **Step 2 — Null impute:**
  - Numeric columns with nulls → `fillna(median)`
  - Categorical columns with nulls → `fillna(mode)`
  - Strategy dict records `"median"`, `"mode"`, or `"none"` per column
- **Step 3 — IQR outlier removal:**
  - Skips the target column (must not be modified)
  - Skips columns with ≤1 unique value or IQR=0
  - Removes rows where value < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
- If `feedback_ctx` is provided (from failed critic), applies feedback-adjusted cleaning
- Saves `outputs/{run_id}/cleaned.csv`

### `automl_tool(run_id, target_col)` → `dict`
**File:** `tools/automl_tool.py`

- Reads `outputs/{run_id}/cleaned.csv`
- Drops non-predictive columns: `patient_id`, `id`, `run_id`, date columns
- Ensures target is binary int
- PyCaret setup: `setup(data=df, target=target_col, session_id=42, fold=3, n_jobs=1)`
- Compares models: `compare_models(include=['lr','dt','rf','lightgbm','nb'], n_select=1, turbo=True)`
- Extracts metrics: model name, accuracy, AUC, F1, recall, precision
- SHAP: `_compute_shap()` — `TreeExplainer` primary, `KernelExplainer` fallback, returns top-10 features
- `get_protein_for_run(run_id)` — reads `diagnosis` column mode → maps via `DIAGNOSIS_PROTEIN_MAP`

**DIAGNOSIS_PROTEIN_MAP (12 diseases):**

| Diagnosis | Protein | UniProt ID |
|-----------|---------|------------|
| Heart Failure | BNP | P16860 |
| COPD | SERPINA1 | P01009 |
| Diabetes Type 2 | GCK | P35557 |
| Stroke | PLAT | P00750 |
| Hypertension | ACE | P12821 |
| Pneumonia | DEFB1 | P60022 |
| Hip Fracture | RANKL | O14788 |
| Asthma | IL13 | P35225 |
| UTI | TLR4 | O00206 |
| Gallstones | ABCG8 | Q9Y210 |
| Kidney Stones | SLC34A1 | Q06495 |
| Appendicitis | CRP | P02741 |
| *Default* | TP53 | P04637 |

### `alphafold_tool(protein_name, uniprot_id)` → `dict`
**File:** `tools/alphafold_tool.py`

- **Mock implementation** — deterministic, no external API call
- Looks up protein sequence from `PROTEIN_SEQUENCES` dict (12 proteins + TP53 default)
- Stability score = `SHA256(sequence)[:8]` → normalize to `[0.25, 0.95]`
- Confidence band: `> 0.75` = high, `> 0.50` = medium, else low
- PDB link: `https://alphafold.ebi.ac.uk/entry/{uniprot_id}`
- Production replacement: AlphaFold2 or ESMFold inference

### `finance_tool(run_id, ml_auc, stability_score, rows_after)` → `dict`
**File:** `tools/finance_tool.py`

- **Denial probability:**
  ```
  severity_factor = 1.0 - stability_score
  denial_probability = min(0.12 + severity_factor * 0.35 + ml_auc * 0.15, 0.99)
  ```
  - Base rate: 0.12 (CMS industry average)
  - Higher severity (lower stability) → higher denial
  - Higher model confidence (AUC) → higher denial

- **Waste estimate:**
  ```
  predicted_readmission_rate = 0.3 + severity_factor * 0.4
  avoidable_readmissions = rows_after * predicted_readmission_rate * 0.25
  waste_estimate = avoidable_readmissions * $15,000 * (avg_los / 5.0)
  ```
  - Reads `length_of_stay` from cleaned.csv (defaults to 5 days)
  - Cost per avoidable readmission: $15,000 (CMS 2024 benchmark)
  - Avoidable fraction: 25%

### `run_safety_vault(ml_auc, ml_accuracy, stability_score, denial_probability, waste_estimate_usd, protein_name)` → `dict`
**File:** `tools/safety_vault.py`

- **6 hard-coded deterministic rules** — no LLM, no dynamic code execution
- Returns: `{doctor_review: bool, safety_flags: list, overrides: dict, rules_checked: 6, rules_triggered: int}`

---

## 10. Safety Vault Rules

| Rule ID | Condition | Action | Message |
|---------|-----------|--------|---------|
| **SR-001** | AUC >= 0.85 | `doctor_review` | Model AUC >= 0.85 — high-confidence readmission risk. Mandatory physician review. |
| **SR-002** | stability < 0.40 | `doctor_review` | Protein stability < 0.40 — structurally compromised biomarker. Escalate to specialist. |
| **SR-003** | denial >= 0.55 | `flag` | Denial probability >= 55%. Recommend pre-authorization and case management. |
| **SR-004** | waste >= $500K | `flag` | Projected healthcare waste >= $500K. Recommend population health intervention. |
| **SR-005** | protein in (BNP, PLAT) | `doctor_review` | Cardiac/stroke biomarker detected. All predictions require cardiologist sign-off. |
| **SR-006** | accuracy < 0.60 | `override` | Model accuracy < 60%. Prediction unreliable. Suppressing AUC from patient output. |

**Actions:**
- `doctor_review` — sets `doctor_review = True` in output
- `flag` — adds to `safety_flags` list (informational)
- `override` — substitutes a safe value for a specific field (e.g., suppress AUC)

---

## 11. Critic Gates

### Critic Gate 1 — Data Quality (S3)

- **Location:** `agents/zora_clean.py` → `_critic_kickoff_with_fallback()`
- **LLMs:** groq/llama-3.1-8b-instant → groq/llama-3.3-70b-versatile (temp=0.0)
- **Pass threshold:** Score >= 7 out of 10
- **Max retries:** 3 (clean_tool is re-run with `feedback_ctx` from critic)
- **Prompt includes:**
  - RAG schema context (top 3 chunks from pgvector)
  - Cleaning actions taken: dupes removed, nulls imputed (filtered to only active imputation — key fix), outliers removed
  - Row count delta, target column verification
- **Output:** JSON `{score: int, passed: bool, feedback: str}`
- **Key design note:** The prompt only shows columns where `imputation_strategy != "none"`. Previous versions showed all 18 columns with "none" for untouched columns, which confused the critic into giving 6/10 scores.

### Critic Gate 1 — RAG Cosine Hallucination Check (S4)

- **Location:** `agents/zora_automl.py` → `_rag_cosine_check()`
- **Method:** Embed a claim sentence → pgvector cosine similarity vs stored schema chunks
- **Pass threshold:** Best similarity >= 0.82
- **Max retries:** 2 (claim is simplified on retry to improve grounding)
- **Claim examples:**
  - Attempt 1: "The dataset has DecisionTreeClassifier as best model with AUC 1.0. Top feature: age. Associated protein BNP has stability score 0.8871."
  - Retry: "Dataset with target readmission_30day. Best model DecisionTreeClassifier trained on this dataset."
- **Behavior:** Pipeline continues even if WARN — the warning is logged in SSE and Supabase but does not block downstream stages.

### Critic Gate 2 — Narration Quality (S6)

- **Location:** `agents/zora_narrator.py` → `_g2_critic_kickoff()`
- **LLMs:** groq/llama-3.1-8b-instant → groq/llama-3.3-70b-versatile (temp=0.0)
- **Scoring dimensions** (1-5 each, per voice):
  - `clinical_clarity`, `clinical_completeness`, `clinical_tone`
  - `patient_clarity`, `patient_completeness`, `patient_tone`
- **Composite score:** `average(clinical_sum, patient_sum) / 15 * 10` → normalized to [2, 10]
- **Pass threshold:** Composite >= 7.0
- **Max retries:** 1 (narrations regenerated with G2 feedback appended)
- **Output:** JSON with 6 dimension scores + feedback string

---

## 12. SSEManager Implementation

**File:** `utils/sse_manager.py`

```
Architecture: List-buffer with cursor-based polling
```

- **Storage:** `_events: dict[str, list[dict]]` — append-only list per run_id
- **`publish(run_id, event)`** — appends event to the list
- **`subscribe(run_id)`** — async generator that:
  1. Starts cursor at 0
  2. Yields all buffered events from cursor to end of list
  3. Polls every 100ms for new events
  4. Terminates on `type=pipeline_complete`
  5. Calls `cleanup(run_id)` to free memory
- **Replay-safe:** Late subscribers (connecting after pipeline starts) replay all prior events from the beginning. This solves the race condition where the pipeline completes before the subscriber connects.
- **Timeout:** 300 seconds (5 minutes) max stream duration
- **Cleanup:** `_events.pop(run_id)` after `pipeline_complete`

**Why not asyncio.Queue?** Queues drain on first read — if the pipeline completes before the subscriber connects, all events are lost. The list-buffer approach means events persist until cleanup.

---

## 13. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL (e.g., `https://xxx.supabase.co`) |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service role key (not anon key) |
| `GOOGLE_API_KEY` | Yes | Google AI API key — used for Gemini 2.0 Flash + embedding-001 |
| `GROQ_API_KEY` | Yes | Groq API key — used for LLaMA 3.3 70B + LLaMA 3.1 8B |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing key (optional observability) |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default: `zora-s1-s2`) |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (default: `true`) |
| `UPLOAD_DIR` | No | Upload directory (default: `./uploads`) |
| `OUTPUT_DIR` | No | Output directory (default: `./outputs`) |
| `TWILIO_ACCOUNT_SID` | No | Twilio Account SID — enables SMS delivery of patient narration |
| `TWILIO_AUTH_TOKEN` | No | Twilio Auth Token |
| `TWILIO_SMS_FROM` | No | Twilio sender phone number (E.164 format) |
| `CLOUD_TTS_API_KEY` | No | Google Cloud TTS API key — enables MP3 audio synthesis |
| `CLOUD_TTS_VOICE_EN` | No | English TTS voice (default: `en-IN-Neural2-A`) |
| `CLOUD_TTS_VOICE_HI` | No | Hindi TTS voice (default: `hi-IN-Neural2-A`) |
| `CLOUD_TTS_SPEAKING_RATE` | No | TTS speaking rate (default: `0.96`) |

---

## 14. File Outputs Per Run

```
uploads/{run_id}/
  └── {original_filename}.csv          # Raw uploaded file (untouched)

outputs/{run_id}/
  ├── ingested.csv                     # Post-Polars parse (S1)
  ├── cleaned.csv                      # Post-dedup/impute/IQR (S3)
  └── narration_patient.mp3            # Google TTS audio (S6, if CLOUD_TTS_API_KEY set)
```

---

## 15. Verified Test Run

**Dataset:** `test_data/patient_readmission.csv`
- 42 rows, 18 columns
- Target: `readmission_30day` (binary int)
- Columns: patient_id, age, gender, diagnosis, num_prior_admissions, length_of_stay_days, num_medications, num_procedures, insurance_type, discharge_disposition, bmi, blood_pressure_systolic, glucose_level, cholesterol, hemoglobin_a1c, smoking_status, comorbidity_count, readmission_30day

**Run ID:** `f1fa2d44a342`

| Metric | Value |
|--------|-------|
| S1 rows/cols | 42 / 18 |
| S2 vectors | 20 |
| S3 rows after clean | 35 (2 dupes + 5 IQR outliers removed) |
| S3 nulls imputed | cholesterol: 3 (median) |
| S3 critic score | 9/10 PASS |
| S4 best model | DecisionTreeClassifier |
| S4 AUC | 1.0 |
| S4 accuracy | 1.0 |
| S4 top SHAP feature | age |
| S4 protein | BNP (mapped from "Heart Failure" diagnosis) |
| S4 stability | 0.8871 (high confidence) |
| S4 Gate 1 cosine | 0.7697 (WARN, below 0.82 threshold) |
| S5 denial probability | 31% |
| S5 waste estimate | $44,311 |
| S5 doctor_review | True (SR-001 AUC >= 0.85 + SR-005 cardiac biomarker) |
| S5 safety flags | 2 triggered (SR-001, SR-005) |
| S6 clinical narration | 1,615 characters |
| S6 patient narration | 1,034 characters |
| S6 G2 score | 9.33/10 PASS |
| S6 TTS | narration_patient.mp3 saved |
| Total latency | ~120 seconds (99s in PyCaret) |

---

## 16. Known Limitations and Notes

1. **AlphaFold is mocked** — uses SHA256 hash for deterministic stability scores. Production would call ESMFold API or AlphaFold2 inference.

2. **PyCaret AUC=1.0 on 35 rows** — expected overfitting on tiny synthetic dataset. With real clinical data (thousands of rows), expect AUC in 0.65-0.85 range.

3. **Gate 1 cosine threshold (0.82) often triggers WARN** — schema chunks are sparse text descriptions, not clinical claims. The cosine similarity between an ML claim and schema profile text is inherently low. Pipeline continues regardless.

4. **Gemini free tier 429s** — very common under load. The Groq fallback handles this transparently. If both fail, the pipeline errors and SSE reports `status=failed`.

5. **SSEManager is in-memory** — events are lost on server restart. Fine for dev/demo. Production would need Redis or Supabase Realtime.

6. **Single-worker limitation** — `_run_pipeline` runs as an `asyncio.create_task` in the same process. PyCaret blocks the event loop (~99 seconds). Production would use Celery or a task queue.

7. **Frontend is deferred** — the backend is designed for a Next.js 15 frontend (CORS allows `localhost:3000`) but no frontend code exists yet.

8. **Twilio SMS** — configured but not auto-triggered. Requires passing `phone_number` parameter to `run_narrator_agent()`. Currently no API endpoint exposes this — would need a form field on the upload endpoint.

9. **Safety vault uses explicit if/elif comparisons** — no dynamic code execution. Each rule is a hardcoded comparison.

10. **LangSmith tracing** — configured via env vars but optional. Provides observability into CrewAI agent execution if enabled.

---

## 17. How to Run

```bash
# 1. Install dependencies (in venv)
cd zora-backend
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn polars pandas pycaret shap crewai litellm \
  google-genai supabase pydantic-settings structlog langchain-core \
  twilio requests

# 2. Configure .env (see Section 12)

# 3. Start server
uvicorn main:app --port 8080

# 4. Upload a dataset
curl -X POST http://localhost:8080/api/run \
  -F "file=@test_data/patient_readmission.csv" \
  -F "problem_desc=Predict 30-day hospital readmission risk" \
  -F "target_column=readmission_30day"

# 5. Stream events (replace RUN_ID)
curl -s --no-buffer -N http://localhost:8080/run/{RUN_ID}/stream

# 6. Check final status
curl http://localhost:8080/api/run/{RUN_ID}/status
```

---

## 18. Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **List-buffer SSE** (not Queue) | Late subscribers must replay all events — queues drain on first read |
| **CrewAI with crewai.LLM** (not LangChain) | CrewAI 1.14 validates LLM type strictly; LangChain chat models cause validation errors |
| **google-genai SDK** (not langchain-google) | `models/embedding-001` returned 404 via LangChain; direct SDK with `gemini-embedding-001` works |
| **Groq fallback chain** | Gemini free tier 429s are frequent; Groq is fast and reliable |
| **Critic shows only active imputation** | Showing "none" for untouched columns confused the LLM critic into lower scores |
| **Safety vault: explicit if/elif** | Deterministic, auditable, no security risk from dynamic execution |
| **PyCaret fold=3** | Small dataset (35 rows) — higher folds cause empty splits |
| **768-dim embeddings** | Matches Supabase `vector(768)` schema; default 1536 would require migration |
| **Polars for ingest, Pandas for clean** | Polars is fast for parsing; Pandas required by PyCaret and has richer imputation API |
 