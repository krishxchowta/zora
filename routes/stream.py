from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from utils.sse_manager import sse_manager

router = APIRouter()


@router.get("/run/{run_id}/stream")
async def stream_run(run_id: str):
    return StreamingResponse(
        sse_manager.subscribe(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
