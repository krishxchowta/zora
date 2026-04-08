from __future__ import annotations

from pathlib import Path

from models.schemas import OpsReadinessCheck, OpsReadinessReport
from services.supabase_service import get_supabase
from utils.config import settings

REQUIRED_TABLES = (
    "patient_contacts",
    "report_requests",
    "prescriptions",
    "message_deliveries",
)
REQUIRED_COLUMNS = {
    "runs": ("feature_summary",),
    "insights": (
        "doctor_report_text",
        "patient_report_text",
        "final_prescription_text",
        "report_status",
    ),
}


def get_readiness_report() -> OpsReadinessReport:
    checks: list[OpsReadinessCheck] = []
    supabase = get_supabase()
    table_status: dict[str, bool] = {}
    missing_tables: list[str] = []
    missing_columns: list[str] = []

    for table in dict.fromkeys((*REQUIRED_TABLES, *REQUIRED_COLUMNS.keys())):
        ok, detail = _probe_table(supabase, table)
        table_status[table] = ok
        if not ok and table in REQUIRED_TABLES:
            missing_tables.append(table)
        checks.append(
            OpsReadinessCheck(
                name=f"table:{table}",
                ok=ok,
                detail=detail,
            )
        )

    for table, columns in REQUIRED_COLUMNS.items():
        if not table_status.get(table, False):
            missing_columns.extend([f"{table}.{column}" for column in columns])
            checks.append(
                OpsReadinessCheck(
                    name=f"columns:{table}",
                    ok=False,
                    detail=f"Skipped column probes because the '{table}' table is unavailable.",
                    missing=[f"{table}.{column}" for column in columns],
                )
            )
            continue

        for column in columns:
            ok, detail = _probe_column(supabase, table, column)
            if not ok:
                missing_columns.append(f"{table}.{column}")
            checks.append(
                OpsReadinessCheck(
                    name=f"column:{table}.{column}",
                    ok=ok,
                    detail=detail,
                    missing=[] if ok else [f"{table}.{column}"],
                )
            )

    sms_missing = []
    if not settings.TWILIO_ACCOUNT_SID:
        sms_missing.append("TWILIO_ACCOUNT_SID")
    if not settings.TWILIO_AUTH_TOKEN:
        sms_missing.append("TWILIO_AUTH_TOKEN")
    if not settings.TWILIO_SMS_FROM:
        sms_missing.append("TWILIO_SMS_FROM")
    sms_ready = not sms_missing
    checks.append(
        OpsReadinessCheck(
            name="twilio:sms",
            ok=sms_ready,
            detail="Twilio SMS can deliver approved patient reports."
            if sms_ready
            else "Twilio SMS is not fully configured.",
            missing=sms_missing,
        )
    )

    whatsapp_missing = []
    whatsapp_warnings = []
    if not settings.TWILIO_ACCOUNT_SID:
        whatsapp_missing.append("TWILIO_ACCOUNT_SID")
    if not settings.TWILIO_AUTH_TOKEN:
        whatsapp_missing.append("TWILIO_AUTH_TOKEN")
    if not settings.TWILIO_WHATSAPP_FROM:
        whatsapp_missing.append("TWILIO_WHATSAPP_FROM")
    elif settings.TWILIO_WHATSAPP_FROM.startswith("whatsapp:"):
        whatsapp_missing.append("TWILIO_WHATSAPP_FROM")
        whatsapp_warnings.append(
            "Use the raw E.164 sender number without the 'whatsapp:' prefix."
        )
    whatsapp_ready = not whatsapp_missing
    checks.append(
        OpsReadinessCheck(
            name="twilio:whatsapp",
            ok=whatsapp_ready,
            detail="Twilio WhatsApp sender is configured."
            if whatsapp_ready
            else "Twilio WhatsApp is not fully configured.",
            missing=whatsapp_missing,
            warnings=whatsapp_warnings,
        )
    )

    doctor_notification_missing = []
    if not settings.DEFAULT_DOCTOR_WHATSAPP_TO:
        doctor_notification_missing.append("DEFAULT_DOCTOR_WHATSAPP_TO")
    if not settings.DOCTOR_APPROVAL_BASE_URL:
        doctor_notification_missing.append("DOCTOR_APPROVAL_BASE_URL")
    doctor_notification_ready = whatsapp_ready and not doctor_notification_missing
    checks.append(
        OpsReadinessCheck(
            name="doctor-notification",
            ok=doctor_notification_ready,
            detail="Doctor WhatsApp approvals can be notified."
            if doctor_notification_ready
            else "Doctor WhatsApp notifications are not fully configured.",
            missing=doctor_notification_missing,
        )
    )

    database_ready = not missing_tables and not missing_columns
    board_delivery_ready = database_ready and sms_ready
    overall_ready = board_delivery_ready and doctor_notification_ready

    required_manual_steps: list[str] = []
    bundle_path = _migration_bundle_path()
    if not database_ready:
        required_manual_steps.append(
            f"Apply the combined SQL bundle in the Supabase SQL Editor: {bundle_path}"
        )
        required_manual_steps.append(
            "Restart the FastAPI backend after the SQL migration so new schema fields are available."
        )
    if not sms_ready:
        required_manual_steps.append(
            "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_SMS_FROM to validate patient SMS delivery."
        )
    if not whatsapp_ready:
        required_manual_steps.append(
            "Finish Twilio WhatsApp Self Sign-up and set TWILIO_WHATSAPP_FROM to the raw E.164 sender number."
        )
    if not doctor_notification_ready:
        required_manual_steps.append(
            "Set DEFAULT_DOCTOR_WHATSAPP_TO and DOCTOR_APPROVAL_BASE_URL so doctor approval alerts can deep-link back into the board."
        )

    return OpsReadinessReport(
        overall_ready=overall_ready,
        database_ready=database_ready,
        sms_ready=sms_ready,
        whatsapp_ready=whatsapp_ready,
        doctor_notification_ready=doctor_notification_ready,
        board_delivery_ready=board_delivery_ready,
        migration_bundle_path=bundle_path,
        checks=checks,
        required_manual_steps=required_manual_steps,
    )


def _probe_table(supabase, table: str) -> tuple[bool, str]:
    try:
        supabase.table(table).select("*").limit(1).execute()
        return True, f"Table '{table}' is reachable."
    except Exception as exc:  # pragma: no cover - provider/network specific
        return False, _clean_error(exc)


def _probe_column(supabase, table: str, column: str) -> tuple[bool, str]:
    try:
        supabase.table(table).select(column).limit(1).execute()
        return True, f"Column '{table}.{column}' is reachable."
    except Exception as exc:  # pragma: no cover - provider/network specific
        return False, _clean_error(exc)


def _clean_error(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _migration_bundle_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "sql"
        / "2026-04-08_last_mile_bundle.sql"
    )
