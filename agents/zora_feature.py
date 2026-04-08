import time
from datetime import datetime, timezone

from models.schemas import FeatureEngineeringReport, SchemaProfile
from services.supabase_service import update_run_status
from tools.feature_engineering_tool import feature_engineering_tool
from utils.sse_manager import sse_manager


async def run_feature_agent(
    run_id: str,
    profile: SchemaProfile,
) -> FeatureEngineeringReport:
    t0 = time.monotonic()

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_feature",
        "status": "running",
        "output_summary": "Engineering clinically interpretable model features...",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    report = feature_engineering_tool(
        run_id=run_id,
        target_col=profile.target_candidate,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    update_run_status(run_id, feature_summary=report.model_dump())

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_feature",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"Feature matrix ready: {report.feature_columns} columns from "
            f"{report.source_columns} cleaned columns. "
            f"{len(report.derived_features_added)} derived features added."
        ),
        "data": report.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return report
