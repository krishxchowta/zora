from fastapi import APIRouter, HTTPException, Query

from models.schemas import (
    MessageSendRequest,
    PatientContactCreateRequest,
    PrescriptionUpsertRequest,
    ReportApprovalRequest,
    ReportRejectRequest,
    ReportRequestNotifyRequest,
)
from services.clinical_board_service import (
    approve_report_request,
    create_patient_report_request,
    get_board_case_detail,
    list_board_cases,
    list_pending_report_requests,
    notify_doctor,
    reject_report_request,
    send_report_request_message,
    upsert_prescription,
)

router = APIRouter()


@router.get("/board/cases")
async def get_board_cases():
    return {"cases": [case.model_dump() for case in list_board_cases()]}


@router.get("/board/cases/{run_id}")
async def get_board_case(run_id: str):
    detail = get_board_case_detail(run_id)
    return detail.model_dump()


@router.post("/board/cases/{run_id}/prescription")
async def save_case_prescription(
    run_id: str,
    payload: PrescriptionUpsertRequest,
    report_request_id: int | None = Query(default=None),
):
    prescription = upsert_prescription(
        run_id=run_id,
        doctor_name=payload.doctor_name,
        prescription_text=payload.prescription_text,
        notes=payload.notes,
        report_request_id=report_request_id,
        is_final=False,
    )
    return {"prescription": prescription}


@router.post("/report-requests")
async def create_report_request(payload: PatientContactCreateRequest):
    return create_patient_report_request(payload.model_dump())


@router.get("/report-requests/pending")
async def get_pending_report_requests():
    return {"report_requests": list_pending_report_requests()}


@router.post("/report-requests/{report_request_id}/notify-doctor")
async def post_notify_doctor(
    report_request_id: int,
    payload: ReportRequestNotifyRequest,
):
    return notify_doctor(
        report_request_id=report_request_id,
        doctor_name=payload.doctor_name,
    )


@router.post("/report-requests/{report_request_id}/approve")
async def post_approve_report_request(
    report_request_id: int,
    payload: ReportApprovalRequest,
):
    return approve_report_request(
        report_request_id=report_request_id,
        doctor_name=payload.doctor_name,
        prescription_text=payload.prescription_text,
        notes=payload.notes,
        send_channel=payload.send_channel,
    )


@router.post("/report-requests/{report_request_id}/reject")
async def post_reject_report_request(
    report_request_id: int,
    payload: ReportRejectRequest,
):
    if not payload.reason.strip():
        raise HTTPException(status_code=400, detail="A rejection reason is required.")
    return reject_report_request(
        report_request_id=report_request_id,
        doctor_name=payload.doctor_name,
        reason=payload.reason,
    )


@router.post("/report-requests/{report_request_id}/send-sms")
async def post_send_sms(
    report_request_id: int,
    payload: MessageSendRequest,
):
    return send_report_request_message(
        report_request_id=report_request_id,
        channel="sms",
        doctor_name=payload.doctor_name,
        notes=payload.notes,
    )


@router.post("/report-requests/{report_request_id}/send-whatsapp")
async def post_send_whatsapp(
    report_request_id: int,
    payload: MessageSendRequest,
):
    return send_report_request_message(
        report_request_id=report_request_id,
        channel="whatsapp",
        doctor_name=payload.doctor_name,
        notes=payload.notes,
    )
