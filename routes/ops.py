from fastapi import APIRouter

from services.ops_service import get_readiness_report

router = APIRouter()


@router.get("/ops/readiness")
async def get_ops_readiness():
    return get_readiness_report().model_dump()
