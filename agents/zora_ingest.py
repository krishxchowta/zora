import time
import os
from crewai import Agent, Task, Crew, Process, LLM
from tools.ingest_tool import ingest_tool
from services.supabase_service import update_run_status
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import SchemaProfile
from datetime import datetime, timezone
import json


def _make_crew(llm: LLM, profile_json: str, target_hint: str | None) -> Crew:
    agent = Agent(
        role="Senior Data Engineer",
        goal=(
            "Review the schema profile of the uploaded dataset. "
            "Confirm or correct the target column candidate. "
            "Flag any data quality concerns. "
            "Return a brief validation note as plain text."
        ),
        backstory=(
            "You are meticulous about data quality. "
            "You always verify schema assumptions before "
            "allowing any downstream analysis."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )
    task = Task(
        description=(
            f"Schema profile:\n{profile_json}\n\n"
            f"Problem description: {target_hint or 'not provided'}\n\n"
            "In 2 sentences: confirm the target column candidate "
            "and flag any obvious data quality issues."
        ),
        expected_output=(
            "2-sentence validation note. "
            "Example: 'Target column confirmed as churn. "
            "income column has 18% null rate — will require imputation.'"
        ),
        agent=agent
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def _kickoff_with_fallback(profile_json: str, target_hint: str | None) -> str:
    """Try Gemini 2.0 Flash first; fall back to Groq llama-3.3-70b on any error."""
    candidates = [
        LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=0.1),
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0.1),
    ]
    last_exc: Exception | None = None
    for llm in candidates:
        try:
            return str(_make_crew(llm, profile_json, target_hint).kickoff())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All LLMs failed. Last error: {last_exc}")


async def run_ingest_agent(
    run_id: str,
    filepath: str,
    target_column: str | None
) -> SchemaProfile:

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_ingest",
        "status": "running",
        "output_summary": "Parsing dataset with Polars...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    t0 = time.monotonic()

    # Run ingest tool directly (no LLM needed for parsing)
    profile = ingest_tool(
        filepath=filepath,
        run_id=run_id,
        target_column=target_column
    )

    # Use LLM to validate schema profile — Gemini primary, Groq fallback
    validation_note = _kickoff_with_fallback(
        profile_json=json.dumps(profile.model_dump(), indent=2),
        target_hint=target_column
    )

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Update Supabase runs table
    update_run_status(
        run_id,
        status="ingested",
        rows_count=profile.rows,
        cols_count=profile.cols,
        schema_json=profile.model_dump()
    )

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_ingest",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"{profile.rows} rows, {profile.cols} cols. "
            f"Target: {profile.target_candidate}. "
            f"{validation_note}"
        ),
        "data": {
            "rows": profile.rows,
            "cols": profile.cols,
            "target_candidate": profile.target_candidate,
            "numeric_cols": len(profile.numeric_columns),
            "categorical_cols": len(profile.categorical_columns),
            "null_cols": len(profile.null_summary)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return profile
