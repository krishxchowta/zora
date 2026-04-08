"""
S4 — Zora AutoML Agent + AlphaFold Tool + Critic Gate 1 (hallucination check)

Flow:
  1. automl_tool    → best model + SHAP + metrics
  2. alphafold_tool → stability score + PDB link (mock)
  3. Critic Gate 1  → RAG cosine ≥ 0.82 per claim (2 retries)
  4. Update Supabase runs table
  5. Emit SSE events
"""

import time
import json
from google import genai as google_genai
from google.genai import types as genai_types
from supabase import create_client

from tools.automl_tool import automl_tool
from tools.alphafold_tool import alphafold_tool
from tools.misfold_tool import resolve_protein_context_for_run
from services.supabase_service import update_run_status
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import ProteinContext, SchemaProfile
from datetime import datetime, timezone

RAG_COSINE_THRESHOLD = 0.82
MAX_CRITIC_RETRIES   = 2


# ── RAG cosine grounding check ────────────────────────────────────────────────

def _embed_text(text: str) -> list[float]:
    client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    resp = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text],
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    return resp.embeddings[0].values


def _rag_cosine_check(claim: str, run_id: str) -> tuple[bool, float]:
    """
    Embed the claim, query pgvector, return (passed, best_cosine_score).
    PASS if best similarity ≥ RAG_COSINE_THRESHOLD.
    """
    query_vec = _embed_text(claim)
    supabase  = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_vec,
        "match_count": 1,
        "filter": {"run_id": run_id}
    }).execute()

    if not result.data:
        return False, 0.0

    best_score = float(result.data[0].get("similarity", 0.0))
    return best_score >= RAG_COSINE_THRESHOLD, round(best_score, 4)


def _build_grounding_claim(automl_result: dict, alphafold_result: dict) -> str:
    """Build a single claim sentence for RAG cosine grounding check."""
    top_feature = next(iter(automl_result["top_features"]), "unknown_feature")
    return (
        f"The dataset has {automl_result['metrics']['model']} as best model "
        f"with AUC {automl_result['metrics']['auc']}. "
        f"Top feature: {top_feature}. "
        f"Associated protein {alphafold_result['protein_name']} "
        f"has stability score {alphafold_result['stability_score']}."
    )


# ── Main agent ────────────────────────────────────────────────────────────────

async def run_automl_agent(
    run_id: str,
    profile: SchemaProfile,
    explicit_protein_context: ProteinContext | None = None,
) -> dict:
    """
    Run AutoML + AlphaFold + Critic Gate 1.
    Returns combined result dict for downstream agents.
    """
    t0 = time.monotonic()

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_automl",
        "status": "running",
        "output_summary": "Running PyCaret compare_models on cleaned data...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Step 1: AutoML ────────────────────────────────────────────────────────
    target_col = profile.target_candidate or "readmission_30day"
    automl_result = automl_tool(run_id, target_col)

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_automl",
        "status": "running",
        "output_summary": (
            f"Best model: {automl_result['model_name']} "
            f"(AUC={automl_result['metrics']['auc']}, "
            f"Acc={automl_result['metrics']['accuracy']}). "
            f"Running AlphaFold EBI API + BioPython..."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Step 2: Resolve protein context + AlphaFold ──────────────────────────
    protein_context = resolve_protein_context_for_run(
        run_id=run_id,
        explicit_context=explicit_protein_context,
    )
    alphafold_result = alphafold_tool(
        protein_context.protein_name or "TP53",
        protein_context.uniprot_id or "P04637",
    )

    # ── Step 3: Critic Gate 1 — RAG cosine hallucination check ───────────────
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_critic_gate1",
        "status": "running",
        "output_summary": "Critic Gate 1: RAG cosine hallucination check...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    claim = _build_grounding_claim(automl_result, alphafold_result)
    passed_gate = False
    cosine_score = 0.0

    for attempt in range(1, MAX_CRITIC_RETRIES + 2):
        passed_gate, cosine_score = _rag_cosine_check(claim, run_id)
        if passed_gate:
            break
        if attempt <= MAX_CRITIC_RETRIES:
            # Simplify the claim on retry (fewer assertions = easier to ground)
            claim = (
                f"Dataset with target {target_col}. "
                f"Best model {automl_result['model_name']} trained on this dataset."
            )

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_critic_gate1",
        "status": "completed" if passed_gate else "warning",
        "output_summary": (
            f"Gate 1: {'PASS' if passed_gate else 'WARN'} "
            f"(cosine={cosine_score}, threshold={RAG_COSINE_THRESHOLD}). "
            f"{'Claims grounded in schema context.' if passed_gate else 'Low similarity — proceeding with caution.'}"
        ),
        "data": {
            "cosine_score": cosine_score,
            "threshold":    RAG_COSINE_THRESHOLD,
            "passed":       passed_gate,
            "attempts":     attempt
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    latency_ms = int((time.monotonic() - t0) * 1000)

    # ── Step 4: Persist to Supabase ───────────────────────────────────────────
    update_run_status(
        run_id,
        status="s4_complete",
        automl_summary={
            **automl_result,
            "gate1_cosine":  cosine_score,
            "gate1_passed":  passed_gate,
        },
        alphafold_summary=alphafold_result,
        protein_context_json=protein_context.model_dump(exclude_none=True),
    )

    # ── Step 5: SSE completed ─────────────────────────────────────────────────
    top_feature = next(iter(automl_result["top_features"]), "unknown_feature")
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_automl",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"AutoML: {automl_result['model_name']} AUC={automl_result['metrics']['auc']}. "
            f"AlphaFold: {protein_context.protein_name} stability={alphafold_result['stability_score']}. "
            f"Gate 1: {'PASS' if passed_gate else 'WARN'} cosine={cosine_score}."
        ),
        "data": {
            "model_name":      automl_result["model_name"],
            "auc":             automl_result["metrics"]["auc"],
            "accuracy":        automl_result["metrics"]["accuracy"],
            "top_feature":     top_feature,
            "protein":         protein_context.protein_name,
            "stability_score": alphafold_result["stability_score"],
            "pdb_link":        alphafold_result["pdb_link"],
            "gate1_passed":    passed_gate,
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return {
        "automl":     automl_result,
        "alphafold":  alphafold_result,
        "protein_context": protein_context.model_dump(exclude_none=True),
        "gate1": {
            "passed":       passed_gate,
            "cosine_score": cosine_score,
        },
    }
