from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException

from models.schemas import BoardCaseDetail, BoardCaseSummary
from services.messaging_service import (
    is_valid_e164,
    send_sms_message,
    send_whatsapp_message,
)
from services.supabase_service import (
    fetch_rows,
    fetch_single,
    get_insight_by_run,
    get_run,
    insert_row,
    update_insight_by_run,
    update_row,
)
from utils.config import settings


def list_board_cases() -> list[BoardCaseSummary]:
    runs = fetch_rows("runs", order_by="created_at", ascending=False)
    cases: list[BoardCaseSummary] = []
    for run in runs:
        run_id = run["run_id"]
        insight = get_insight_by_run(run_id)
        report_requests = fetch_rows(
            "report_requests",
            filters={"run_id": run_id},
            order_by="requested_at",
            ascending=False,
        )
        patient_contacts = fetch_rows(
            "patient_contacts",
            filters={"run_id": run_id},
            order_by="created_at",
            ascending=False,
        )
        deliveries = fetch_rows(
            "message_deliveries",
            filters={"run_id": run_id},
            order_by="sent_at",
            ascending=False,
        )

        latest_request = report_requests[0] if report_requests else None
        latest_delivery = deliveries[0] if deliveries else None
        latest_contact = _find_contact_for_request(latest_request, patient_contacts) if latest_request else None
        if not latest_contact and patient_contacts:
            latest_contact = patient_contacts[0]

        cases.append(BoardCaseSummary(
            run_id=run_id,
            filename=run.get("filename", "unknown.csv"),
            pipeline_status=run.get("status", "unknown"),
            request_status=latest_request.get("status") if latest_request else None,
            message_status=latest_delivery.get("delivery_status") if latest_delivery else None,
            patient_name=(latest_contact or {}).get("patient_name"),
            doctor_review=bool((insight or {}).get("doctor_review", False)),
            doctor_report_ready=bool(_doctor_report_text(insight)),
            patient_report_ready=bool(_patient_report_text(insight)),
            created_at=run.get("created_at"),
            updated_at=_case_updated_at(run, insight, latest_request, latest_delivery),
        ))
    return cases


def get_board_case_detail(run_id: str) -> BoardCaseDetail:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    insight = get_insight_by_run(run_id)
    patient_contacts = fetch_rows(
        "patient_contacts",
        filters={"run_id": run_id},
        order_by="created_at",
        ascending=False,
    )
    report_requests = fetch_rows(
        "report_requests",
        filters={"run_id": run_id},
        order_by="requested_at",
        ascending=False,
    )
    for report_request in report_requests:
        report_request["patient_contact"] = _find_contact_for_request(
            report_request,
            patient_contacts,
        )
    prescriptions = fetch_rows(
        "prescriptions",
        filters={"run_id": run_id},
        order_by="updated_at",
        ascending=False,
    )
    message_deliveries = fetch_rows(
        "message_deliveries",
        filters={"run_id": run_id},
        order_by="sent_at",
        ascending=False,
    )

    return BoardCaseDetail(
        run=run,
        insight=insight,
        patient_contacts=patient_contacts,
        report_requests=report_requests,
        prescriptions=prescriptions,
        message_deliveries=message_deliveries,
        doctor_report_text=_doctor_report_text(insight),
        patient_report_text=_patient_report_text(insight),
        final_prescription_text=(insight or {}).get("final_prescription_text"),
    )


def upsert_prescription(
    run_id: str,
    doctor_name: str,
    prescription_text: str,
    notes: str | None = None,
    report_request_id: int | None = None,
    is_final: bool = False,
) -> dict:
    _require_run(run_id)
    filters = {
        "run_id": run_id,
        "is_final": is_final,
    }
    if report_request_id is not None:
        filters["report_request_id"] = report_request_id
    existing = fetch_rows(
        "prescriptions",
        filters=filters,
        order_by="updated_at",
        ascending=False,
    )
    payload = {
        "doctor_name": doctor_name,
        "prescription_text": prescription_text,
        "notes": notes,
        "updated_at": _now(),
    }
    if existing:
        return update_row(
            "prescriptions",
            "id",
            existing[0]["id"],
            payload,
        ) or existing[0]

    payload.update({
        "run_id": run_id,
        "report_request_id": report_request_id,
        "is_final": is_final,
        "created_at": _now(),
    })
    return insert_row("prescriptions", payload) or payload


def create_patient_report_request(payload: dict) -> dict:
    run_id = payload["run_id"]
    _require_run(run_id)
    phone_e164 = payload.get("phone_e164")
    whatsapp_e164 = payload.get("whatsapp_e164")
    preferred_channel = payload.get("preferred_channel", "whatsapp")

    if not phone_e164 and not whatsapp_e164:
        raise HTTPException(
            status_code=400,
            detail="At least one patient phone number is required.",
        )
    if phone_e164 and not is_valid_e164(phone_e164):
        raise HTTPException(
            status_code=400,
            detail="phone_e164 must be a valid E.164 number.",
        )
    if whatsapp_e164 and not is_valid_e164(whatsapp_e164):
        raise HTTPException(
            status_code=400,
            detail="whatsapp_e164 must be a valid E.164 number.",
        )
    if preferred_channel == "sms" and not phone_e164:
        raise HTTPException(
            status_code=400,
            detail="A preferred SMS delivery requires phone_e164.",
        )
    if preferred_channel == "whatsapp" and not whatsapp_e164:
        raise HTTPException(
            status_code=400,
            detail="A preferred WhatsApp delivery requires whatsapp_e164.",
        )

    contact = insert_row("patient_contacts", {
        "run_id": run_id,
        "patient_name": payload["patient_name"],
        "phone_e164": phone_e164,
        "whatsapp_e164": whatsapp_e164,
        "preferred_channel": preferred_channel,
        "created_at": _now(),
    })
    if not contact:
        raise HTTPException(status_code=500, detail="Failed to create patient contact")

    report_request = insert_row("report_requests", {
        "run_id": run_id,
        "patient_contact_id": contact["id"],
        "request_message": payload.get("request_message"),
        "status": "requested",
        "requested_at": _now(),
    })
    if not report_request:
        raise HTTPException(status_code=500, detail="Failed to create report request")

    notify_result = notify_doctor(report_request["id"])
    report_request["notify_result"] = notify_result
    return report_request


def notify_doctor(report_request_id: int, doctor_name: str | None = None) -> dict:
    report_request = _require_report_request(report_request_id)
    run = _require_run(report_request["run_id"])

    if not settings.DEFAULT_DOCTOR_WHATSAPP_TO:
        result = {
            "ok": False,
            "channel": "whatsapp",
            "delivery_status": "failed",
            "error_text": "DEFAULT_DOCTOR_WHATSAPP_TO is not configured.",
        }
    else:
        deep_link = f"{settings.DOCTOR_APPROVAL_BASE_URL.rstrip('/')}/cases/{run['run_id']}?requestId={report_request_id}"
        body = (
            f"Doctor alert: patient report request pending for run {run['run_id']} "
            f"({run.get('filename', 'analysis')}). Review here: {deep_link}"
        )
        result = send_whatsapp_message(
            run_id=run["run_id"],
            report_request_id=report_request_id,
            recipient_role="doctor",
            recipient_e164=settings.DEFAULT_DOCTOR_WHATSAPP_TO,
            message_type="doctor_report_request",
            body=body,
        )

    update_row(
        "report_requests",
        "id",
        report_request_id,
        {
            "status": (
                report_request["status"]
                if report_request["status"] in {"approved", "sent", "rejected"}
                else "doctor_notified"
            ),
            "doctor_name": doctor_name,
            "doctor_notified_at": _now(),
        },
    )
    return result


def approve_report_request(
    report_request_id: int,
    doctor_name: str,
    prescription_text: str,
    notes: str | None = None,
    send_channel: str = "preferred",
) -> dict:
    report_request = _require_report_request(report_request_id)
    if report_request["status"] in {"approved", "sent"}:
        raise HTTPException(status_code=409, detail="This request has already been approved.")
    if report_request["status"] == "rejected":
        raise HTTPException(status_code=409, detail="Rejected requests cannot be approved.")

    run_id = report_request["run_id"]
    insight = _require_insight(run_id)
    patient_report = _patient_report_text(insight)
    if not patient_report:
        raise HTTPException(status_code=409, detail="Patient report draft is not available yet.")

    patient_contact = _require_patient_contact(report_request["patient_contact_id"])
    final_prescription = upsert_prescription(
        run_id=run_id,
        doctor_name=doctor_name,
        prescription_text=prescription_text,
        notes=notes,
        report_request_id=report_request_id,
        is_final=True,
    )

    update_row(
        "report_requests",
        "id",
        report_request_id,
        {
            "status": "approved",
            "doctor_name": doctor_name,
            "approved_at": _now(),
        },
    )
    update_insight_by_run(
        run_id,
        final_prescription_text=prescription_text,
        report_status="approved",
    )

    delivery = _send_patient_package(
        run_id=run_id,
        report_request_id=report_request_id,
        patient_contact=patient_contact,
        patient_report=patient_report,
        prescription_text=prescription_text,
        channel=send_channel,
    )

    if delivery["ok"]:
        update_row(
            "report_requests",
            "id",
            report_request_id,
            {
                "status": "sent",
                "sent_at": _now(),
            },
        )
        update_insight_by_run(run_id, report_status="sent")

    return {
        "report_request_id": report_request_id,
        "status": "sent" if delivery["ok"] else "approved",
        "delivery": delivery,
        "prescription": final_prescription,
    }


def reject_report_request(report_request_id: int, doctor_name: str, reason: str) -> dict:
    report_request = _require_report_request(report_request_id)
    if report_request["status"] == "sent":
        raise HTTPException(status_code=409, detail="Sent requests cannot be rejected.")

    updated = update_row(
        "report_requests",
        "id",
        report_request_id,
        {
            "status": "rejected",
            "doctor_name": doctor_name,
            "rejected_at": _now(),
        },
    )
    update_insight_by_run(report_request["run_id"], report_status="rejected")
    return {
        "report_request_id": report_request_id,
        "status": "rejected",
        "doctor_name": doctor_name,
        "reason": reason,
        "report_request": updated,
    }


def send_report_request_message(
    report_request_id: int,
    channel: str,
    doctor_name: str,
    notes: str | None = None,
) -> dict:
    if channel not in {"sms", "whatsapp"}:
        raise HTTPException(status_code=400, detail="Unsupported channel")

    report_request = _require_report_request(report_request_id)
    if report_request["status"] not in {"approved", "sent"}:
        raise HTTPException(
            status_code=409,
            detail="The report must be approved before it can be sent to the patient.",
        )

    insight = _require_insight(report_request["run_id"])
    patient_report = _patient_report_text(insight)
    prescription = insight.get("final_prescription_text")
    if not patient_report or not prescription:
        raise HTTPException(
            status_code=409,
            detail="Approved patient report and prescription are required before delivery.",
        )

    patient_contact = _require_patient_contact(report_request["patient_contact_id"])
    delivery = _send_patient_package(
        run_id=report_request["run_id"],
        report_request_id=report_request_id,
        patient_contact=patient_contact,
        patient_report=patient_report,
        prescription_text=prescription,
        channel=channel,
    )
    if delivery["ok"]:
        update_row(
            "report_requests",
            "id",
            report_request_id,
            {
                "status": "sent",
                "sent_at": _now(),
                "doctor_name": doctor_name,
            },
        )
        update_insight_by_run(report_request["run_id"], report_status="sent")
    return {
        "report_request_id": report_request_id,
        "delivery": delivery,
        "notes": notes,
    }


def list_pending_report_requests() -> list[dict]:
    requests = fetch_rows(
        "report_requests",
        order_by="requested_at",
        ascending=False,
    )
    pending = [
        row for row in requests
        if row.get("status") in {"requested", "doctor_notified", "approved"}
    ]
    for row in pending:
        row["run"] = get_run(row["run_id"])
        row["patient_contact"] = fetch_single("patient_contacts", {"id": row["patient_contact_id"]})
    return pending


def _send_patient_package(
    run_id: str,
    report_request_id: int,
    patient_contact: dict,
    patient_report: str,
    prescription_text: str,
    channel: str,
) -> dict:
    preferred_channel = patient_contact.get("preferred_channel", "whatsapp")
    chosen_channel = preferred_channel if channel == "preferred" else channel
    body = (
        f"{patient_report}\n\nPrescription:\n{prescription_text}"
    )

    if chosen_channel == "sms":
        recipient = patient_contact.get("phone_e164")
        return send_sms_message(
            run_id=run_id,
            report_request_id=report_request_id,
            recipient_role="patient",
            recipient_e164=recipient or "",
            message_type="patient_final_report",
            body=body,
        )

    recipient = patient_contact.get("whatsapp_e164")
    return send_whatsapp_message(
        run_id=run_id,
        report_request_id=report_request_id,
        recipient_role="patient",
        recipient_e164=recipient or "",
        message_type="patient_final_report",
        body=body,
    )


def _require_run(run_id: str) -> dict:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _require_insight(run_id: str) -> dict:
    insight = get_insight_by_run(run_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found for run")
    return insight


def _require_report_request(report_request_id: int) -> dict:
    report_request = fetch_single("report_requests", {"id": report_request_id})
    if not report_request:
        raise HTTPException(status_code=404, detail="Report request not found")
    return report_request


def _require_patient_contact(patient_contact_id: int) -> dict:
    patient_contact = fetch_single("patient_contacts", {"id": patient_contact_id})
    if not patient_contact:
        raise HTTPException(status_code=404, detail="Patient contact not found")
    return patient_contact


def _doctor_report_text(insight: dict | None) -> str | None:
    if not insight:
        return None
    return insight.get("doctor_report_text") or insight.get("narration_clinical")


def _patient_report_text(insight: dict | None) -> str | None:
    if not insight:
        return None
    return insight.get("patient_report_text") or insight.get("narration_patient")


def _find_contact_for_request(report_request: dict | None, patient_contacts: list[dict]) -> dict | None:
    if not report_request:
        return None
    patient_contact_id = report_request.get("patient_contact_id")
    for patient_contact in patient_contacts:
        if patient_contact.get("id") == patient_contact_id:
            return patient_contact
    return None


def _case_updated_at(
    run: dict,
    insight: dict | None,
    report_request: dict | None,
    delivery: dict | None,
) -> str | None:
    candidates = [
        run.get("completed_at"),
        run.get("created_at"),
        (insight or {}).get("created_at"),
        (report_request or {}).get("sent_at"),
        (report_request or {}).get("approved_at"),
        (report_request or {}).get("requested_at"),
        (delivery or {}).get("sent_at"),
    ]
    populated = [value for value in candidates if value]
    return max(populated) if populated else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
