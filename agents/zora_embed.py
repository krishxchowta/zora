import time
from tools.embed_tool import embed_tool
from services.supabase_service import update_run_status
from utils.sse_manager import sse_manager
from models.schemas import SchemaProfile
from datetime import datetime, timezone


async def run_embed_agent(
    run_id: str,
    profile: SchemaProfile
) -> int:

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_embed",
        "status": "running",
        "output_summary": "Generating embeddings via gemini-embedding-001 (768-dim)...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    t0 = time.monotonic()

    # Store vectors in Supabase pgvector
    vector_count = embed_tool(profile)

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Update runs table with embedding count
    update_run_status(
        run_id,
        status="embedded",
        embedding_count=vector_count
    )

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_embed",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"{vector_count} vectors stored in Supabase pgvector. "
            f"RAG context ready for downstream agents."
        ),
        "data": {
            "vector_count": vector_count,
            "table": "documents",
            "embedding_model": "models/gemini-embedding-001"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # S2 complete — pipeline_complete is now emitted by S3 (zora_clean)
    return vector_count
