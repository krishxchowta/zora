import time
from datetime import datetime, timezone

from models.schemas import MisfoldSummary, ProteinContext
from services.supabase_service import update_run_status
from tools.misfold_tool import misfold_tool
from utils.sse_manager import sse_manager


async def run_misfold_agent(
    run_id: str,
    s4_result: dict,
    enable_protein_analysis: bool = False,
) -> MisfoldSummary | None:
    if not enable_protein_analysis:
        return None

    protein_context = ProteinContext.model_validate(
        s4_result.get("protein_context", {})
    )

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_misfold",
        "status": "running",
        "output_summary": "Scoring protein aggregation propensity and energy landscape...",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    t0 = time.monotonic()
    summary = misfold_tool(
        run_id=run_id,
        protein_context=protein_context,
        alphafold_result=s4_result.get("alphafold", {}),
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    update_run_status(
        run_id,
        status="s4_complete",
        protein_context_json=protein_context.model_dump(exclude_none=True),
        protein_summary_json=summary.model_dump(),
    )

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_misfold",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"Misfold risk scored for {summary.protein_name}. "
            f"Stuck-score={summary.stuck_score} ({summary.energy_state}). "
            f"Aggregation propensity={summary.aggregation_propensity}."
        ),
        "data": {
            "protein_name": summary.protein_name,
            "variant_hgvs": summary.variant_hgvs,
            "stuck_score": summary.stuck_score,
            "energy_state": summary.energy_state,
            "aggregation_propensity": summary.aggregation_propensity,
            "surface_exposure_score": summary.surface_exposure_score,
            "variant_delta_score": summary.variant_delta_score,
            "red_flags_count": len(summary.red_flags),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return summary
