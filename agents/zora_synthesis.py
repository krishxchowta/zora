"""
S5 — Zora Synthesis Agent

Merges:
  - ML output (model, AUC, SHAP features)
  - AlphaFold stability score + PDB link
  - RAG citations from pgvector
  - Safety vault flags
  - Finance tool output (denial_probability, waste_estimate)

LLM writes a structured synthesis_text.
Writes complete insight row to Supabase `insights` table.
"""

import time
from crewai import Agent, Task, Crew, Process, LLM
from google import genai as google_genai
from google.genai import types as genai_types
from supabase import create_client

from tools.finance_tool import finance_tool
from tools.safety_vault import run_safety_vault
from services.supabase_service import insert_insight_row
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import SchemaProfile, CleanReport
from datetime import datetime, timezone


# ── RAG citation retrieval ────────────────────────────────────────────────────

def _retrieve_rag_citations(run_id: str, query: str, k: int = 5) -> list[dict]:
    client  = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    resp    = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[query],
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    vec     = resp.embeddings[0].values
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    result  = supabase.rpc("match_documents", {
        "query_embedding": vec,
        "match_count": k,
        "filter": {"run_id": run_id}
    }).execute()
    return [
        {"chunk_text": r["chunk_text"], "similarity": round(r["similarity"], 4)}
        for r in (result.data or [])
    ]


# ── LLM synthesis ─────────────────────────────────────────────────────────────

def _synthesis_kickoff(prompt: str) -> str:
    candidates = [
        LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=0.2),
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0.2),
    ]
    last_exc = None
    for llm in candidates:
        try:
            agent = Agent(
                role="Healthcare Analytics Synthesizer",
                goal=(
                    "Merge ML predictions, protein stability data, financial risk indicators, "
                    "and RAG-grounded citations into a concise structured analysis. "
                    "Be factual, cite evidence from the schema context."
                ),
                backstory="Expert in clinical ML, proteomics, and health economics.",
                llm=llm, verbose=False, allow_delegation=False, max_iter=1
            )
            task = Task(
                description=prompt,
                expected_output=(
                    "A structured paragraph (5-8 sentences) covering: "
                    "1) ML model performance, 2) key risk features, "
                    "3) protein stability interpretation, "
                    "4) financial risk, 5) safety flags. "
                    "End with one actionable recommendation."
                ),
                agent=agent
            )
            return str(Crew(agents=[agent], tasks=[task],
                            process=Process.sequential, verbose=False).kickoff())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Synthesis LLM failed: {last_exc}")


# ── Main agent ────────────────────────────────────────────────────────────────

async def run_synthesis_agent(
    run_id: str,
    profile: SchemaProfile,
    clean_report: CleanReport,
    s4_result: dict,
) -> dict:

    t0 = time.monotonic()

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_synthesis",
        "status": "running",
        "output_summary": "Merging ML + AlphaFold + RAG citations via FinanceTool...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    automl    = s4_result["automl"]
    alphafold = s4_result["alphafold"]
    misfold   = s4_result.get("misfold")
    metrics   = automl["metrics"]

    # ── Finance Tool ──────────────────────────────────────────────────────────
    finance = finance_tool(
        run_id=run_id,
        ml_auc=metrics["auc"],
        stability_score=alphafold["stability_score"],
        rows_after=clean_report.rows_after,
    )

    # ── Safety Vault ──────────────────────────────────────────────────────────
    safety = run_safety_vault(
        ml_auc=metrics["auc"],
        ml_accuracy=metrics["accuracy"],
        stability_score=alphafold["stability_score"],
        denial_probability=finance["denial_probability"],
        waste_estimate_usd=finance["waste_estimate_usd"],
        protein_name=alphafold["protein_name"],
        misfold_summary=misfold,
    )

    # ── RAG Citations ─────────────────────────────────────────────────────────
    rag_query = (
        f"Healthcare dataset {profile.filename} with target {profile.target_candidate}. "
        f"Columns: {', '.join(profile.numeric_columns[:5])}."
    )
    rag_citations = _retrieve_rag_citations(run_id, rag_query, k=5)

    # ── LLM Synthesis ─────────────────────────────────────────────────────────
    top_features_str = ", ".join(
        f"{k}={v:.4f}" for k, v in list(automl["top_features"].items())[:5]
    )
    rag_context = "\n".join(c["chunk_text"] for c in rag_citations[:3])
    safety_msg  = "; ".join(f["message"] for f in safety["safety_flags"]) or "No safety flags."
    misfold_section = ""
    if misfold and misfold.get("enabled"):
        variant_delta = misfold.get("variant_delta_score")
        variant_delta_text = (
            f"{variant_delta:.2f}" if isinstance(variant_delta, (int, float))
            else "no curated delta available"
        )
        misfold_evidence = "\n".join(
            f"- {item.get('source')}: {item.get('type')}"
            for item in misfold.get("evidence", [])[:3]
        ) or "- No protein evidence available."
        misfold_section = f"""

MISFOLD RISK:
  Stuck-score: {misfold.get('stuck_score')} ({misfold.get('energy_state')})
  Aggregation propensity: {misfold.get('aggregation_propensity')}
  Surface exposure: {misfold.get('surface_exposure_score')}
  Disorder score: {misfold.get('disorder_score')}
  Variant delta score: {variant_delta_text}
  Evidence:
{misfold_evidence}
""".rstrip()

    synthesis_prompt = f"""
Synthesize a clinical ML analytics report using the data below.

DATASET: {profile.filename} | {clean_report.rows_after} rows | Target: {profile.target_candidate}

ML MODEL: {metrics['model']} | AUC: {metrics['auc']} | Accuracy: {metrics['accuracy']} | F1: {metrics['f1']}
TOP SHAP FEATURES: {top_features_str}

PROTEIN BIOMARKER: {alphafold['protein_name']} (UniProt: {alphafold.get('uniprot_id','')})
STABILITY SCORE: {alphafold['stability_score']} ({alphafold['confidence']} confidence)
PDB LINK: {alphafold['pdb_link']}

FINANCIAL RISK:
  Denial probability: {finance['denial_probability']*100:.1f}%
  Healthcare waste estimate: ${finance['waste_estimate_usd']:,.0f}
  Cohort readmission rate: {finance['predicted_readmission_rate']*100:.0f}%

SAFETY FLAGS: {safety_msg}
DOCTOR REVIEW REQUIRED: {safety['doctor_review']}
{misfold_section}

SCHEMA CONTEXT (RAG):
{rag_context}

Write a structured synthesis paragraph for a clinical audience.
""".strip()

    synthesis_text = _synthesis_kickoff(synthesis_prompt)

    latency_ms = int((time.monotonic() - t0) * 1000)

    # ── Write to Supabase insights table ─────────────────────────────────────
    insight_row = {
        "run_id":              run_id,
        "ml_model":            metrics["model"],
        "ml_accuracy":         metrics["accuracy"],
        "ml_auc":              metrics["auc"],
        "top_features":        automl["top_features"],
        "stability_score":     alphafold["stability_score"],
        "pdb_link":            alphafold["pdb_link"],
        "protein_name":        alphafold["protein_name"],
        "denial_probability":  finance["denial_probability"],
        "waste_estimate":      finance["waste_estimate_usd"],
        "rag_citations":       rag_citations,
        "synthesis_text":      synthesis_text,
        "safety_flags":        safety["safety_flags"],
        "doctor_review":       safety["doctor_review"],
        "protein_summary_json": misfold if misfold and misfold.get("enabled") else None,
    }
    inserted_row = insert_insight_row(insight_row)
    insight_id = inserted_row["id"] if inserted_row else None
    misfold_summary_text = ""
    if misfold and misfold.get("enabled"):
        misfold_summary_text = f"Misfold stuck-score: {misfold.get('stuck_score')}. "

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_synthesis",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"Synthesis complete. Denial risk: {finance['denial_probability']*100:.0f}%. "
            f"Waste estimate: ${finance['waste_estimate_usd']:,.0f}. "
            f"Doctor review: {safety['doctor_review']}. "
            f"{misfold_summary_text}"
            f"Insight #{insight_id} written to Supabase."
        ),
        "data": {
            "insight_id":         insight_id,
            "denial_probability": finance["denial_probability"],
            "waste_estimate_usd": finance["waste_estimate_usd"],
            "doctor_review":      safety["doctor_review"],
            "safety_flags_count": safety["rules_triggered"],
            "rag_citations_count":len(rag_citations),
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return {
        "insight_id":     insight_id,
        "synthesis_text": synthesis_text,
        "finance":        finance,
        "safety":         safety,
        "rag_citations":  rag_citations,
        "misfold":        misfold,
    }
