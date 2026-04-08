import time
import json
from crewai import Agent, Task, Crew, Process, LLM
from google import genai as google_genai
from google.genai import types as genai_types
from supabase import create_client
from tools.clean_tool import clean_tool
from services.supabase_service import update_run_status
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import SchemaProfile, CleanReport
from datetime import datetime, timezone

MAX_CRITIC_RETRIES = 3
PASS_THRESHOLD = 7


# ── RAG RETRIEVAL ─────────────────────────────────────────────────────────────

def _retrieve_schema_context(run_id: str, query: str, k: int = 3) -> str:
    """
    Embed the query using gemini-embedding-001 and retrieve top-k
    schema chunks for this run_id from Supabase pgvector.
    Returns concatenated chunk_text as context string.
    """
    client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[query],
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    query_vector = response.embeddings[0].values

    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_count": k,
        "filter": {"run_id": run_id}
    }).execute()

    if not result.data:
        return ""

    return "\n".join(row["chunk_text"] for row in result.data)


# ── CRITIC LLM ────────────────────────────────────────────────────────────────

def _make_critic_crew(llm: LLM, prompt: str) -> Crew:
    agent = Agent(
        role="Data Quality Critic",
        goal=(
            "Evaluate the quality of a data cleaning operation. "
            "Score from 0 to 10. Score >= 7 means the cleaning is acceptable. "
            "Return ONLY a JSON object with keys: score (int), passed (bool), feedback (str)."
        ),
        backstory=(
            "You are an expert data quality judge. "
            "You verify that cleaning decisions are justified by the schema profile. "
            "You flag hallucinated or unjustified cleaning steps."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )
    task = Task(
        description=prompt,
        expected_output=(
            'Valid JSON object only. Example: '
            '{"score": 8, "passed": true, '
            '"feedback": "Median imputation correct for cholesterol. 2 dupes removed cleanly."}'
        ),
        agent=agent
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def _critic_kickoff_with_fallback(prompt: str) -> dict:
    """
    Run the critic with llama-3.1-8b-instant primary (as per architecture diagram),
    fall back to llama-3.3-70b-versatile. Returns parsed JSON dict.
    """
    candidates = [
        LLM(model="groq/llama-3.1-8b-instant",
            api_key=settings.GROQ_API_KEY, temperature=0.0),
        LLM(model="groq/llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY, temperature=0.0),
    ]
    last_exc: Exception | None = None
    for llm in candidates:
        try:
            raw = str(_make_critic_crew(llm, prompt).kickoff())
            # Extract JSON from response (strip markdown fences if present)
            raw = raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Critic LLM failed after all fallbacks. Last: {last_exc}")


# ── MAIN AGENT ────────────────────────────────────────────────────────────────

async def run_clean_agent(
    run_id: str,
    profile: SchemaProfile
) -> CleanReport:

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_clean",
        "status": "running",
        "output_summary": "Cleaning dataset: null impute + dedup + outlier IQR...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    t0 = time.monotonic()
    feedback_ctx: str | None = None
    clean_report: CleanReport | None = None
    critic_result: dict = {}

    for attempt in range(1, MAX_CRITIC_RETRIES + 1):

        # ── Step 1: Run clean_tool ────────────────────────────────────────────
        clean_report = clean_tool(
            run_id=run_id,
            profile=profile,
            feedback_ctx=feedback_ctx
        )

        # ── Step 2: RAG — retrieve schema context for grounding ───────────────
        rag_query = (
            f"Dataset {profile.filename} schema profile: "
            f"columns, null rates, data types, target column"
        )
        schema_context = _retrieve_schema_context(run_id, rag_query, k=3)

        # ── Step 3: Build critic prompt ───────────────────────────────────────
        await sse_manager.publish(run_id, {
            "type": "agent_update",
            "agent": "zora_critic",
            "status": "running",
            "output_summary": f"Critic evaluating cleaning quality (attempt {attempt}/{MAX_CRITIC_RETRIES})...",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Only show columns where an action was actually taken
        active_imputation = {
            col: strategy
            for col, strategy in clean_report.imputation_strategy.items()
            if strategy != "none"
        }

        critic_prompt = f"""
You are evaluating a data cleaning operation. Score it 0-10 (>=7 = PASS).

SCHEMA CONTEXT (from RAG):
{schema_context or 'No context available.'}

ACTIONS TAKEN:
- Duplicate rows removed: {clean_report.dupes_removed}
- Same-visit duplicates removed: {clean_report.same_visit_dupes_removed}
- Columns with null imputation: {json.dumps(clean_report.nulls_imputed)} (count imputed per col)
- Imputation method used: {json.dumps(active_imputation)} (only cols that had nulls)
- Invalid values converted to null: {json.dumps(clean_report.invalid_values_converted)}
- Missingness flags added: {json.dumps(clean_report.missingness_flags_added)}
- Extreme values capped via winsorization: {json.dumps(clean_report.capped_extremes)}
- Rows: {clean_report.rows_before} → {clean_report.rows_after}
- Target column skipped (correct): {profile.target_candidate}
- Columns with NO nulls were NOT imputed (strategy "none" = no action taken)

{f'PRIOR FEEDBACK (retry {attempt}): {feedback_ctx}' if feedback_ctx else ''}

Score criteria:
1. Median imputation for numeric null columns = good (score up)
2. Mode imputation for categorical null columns = good (score up)
3. Deduplication = always correct (score up)
4. Converting implausible vitals/labs to null with flags = good (score up)
5. Capping extremes instead of dropping rows = good for clinical preservation
5. Target column untouched = correct (score up)

Return ONLY valid JSON: {{"score": <int 0-10>, "passed": <bool>, "feedback": <str>}}
""".strip()

        # ── Step 4: Run critic ────────────────────────────────────────────────
        critic_result = _critic_kickoff_with_fallback(critic_prompt)
        score = int(critic_result.get("score", 0))
        passed = score >= PASS_THRESHOLD

        clean_report.quality_score = score
        clean_report.critic_feedback = critic_result.get("feedback", "")
        clean_report.passed_critic = passed

        await sse_manager.publish(run_id, {
            "type": "agent_update",
            "agent": "zora_critic",
            "status": "completed" if passed else "failed",
            "output_summary": (
                f"Score: {score}/10. {'PASS' if passed else 'FAIL'}. "
                f"{clean_report.critic_feedback}"
            ),
            "data": {
                "score": score,
                "passed": passed,
                "attempt": attempt,
                "feedback": clean_report.critic_feedback
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if passed:
            break

        # ── Retry: send feedback context back to clean_tool ──────────────────
        if attempt < MAX_CRITIC_RETRIES:
            feedback_ctx = clean_report.critic_feedback

    # ── Step 5: Persist to Supabase ───────────────────────────────────────────
    latency_ms = int((time.monotonic() - t0) * 1000)

    update_run_status(
        run_id,
        status="s3_complete",
        cleaned_rows=clean_report.rows_after,
        quality_score=clean_report.quality_score,
        cleaning_summary=clean_report.model_dump(),
        completed_at=datetime.now(timezone.utc).isoformat()
    )

    # ── Step 6: SSE — clean completed ─────────────────────────────────────────
    rows_delta = clean_report.rows_before - clean_report.rows_after
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_clean",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"{clean_report.rows_before}→{clean_report.rows_after} rows. "
            f"{clean_report.dupes_removed + clean_report.same_visit_dupes_removed} dupes removed. "
            f"{sum(clean_report.nulls_imputed.values())} nulls imputed. "
            f"{sum(clean_report.invalid_values_converted.values())} invalid values nulled. "
            f"{sum(clean_report.capped_extremes.values())} extremes capped. "
            f"Quality score: {clean_report.quality_score}/10."
        ),
        "data": {
            "rows_before": clean_report.rows_before,
            "rows_after": clean_report.rows_after,
            "dupes_removed": clean_report.dupes_removed,
            "same_visit_dupes_removed": clean_report.same_visit_dupes_removed,
            "nulls_imputed": clean_report.nulls_imputed,
            "invalid_values_converted": clean_report.invalid_values_converted,
            "capped_extremes": clean_report.capped_extremes,
            "missingness_flags_added": clean_report.missingness_flags_added,
            "quality_score": clean_report.quality_score,
            "passed_critic": clean_report.passed_critic
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return clean_report
