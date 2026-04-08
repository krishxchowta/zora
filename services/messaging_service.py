from __future__ import annotations

import re
from datetime import datetime, timezone

from models.schemas import MessageDeliveryResult
from services.supabase_service import insert_row
from utils.config import settings

E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")


def is_valid_e164(value: str | None) -> bool:
    return bool(value and E164_PATTERN.fullmatch(value))


def send_sms_message(
    run_id: str,
    report_request_id: int | None,
    recipient_role: str,
    recipient_e164: str,
    message_type: str,
    body: str,
) -> dict:
    if not is_valid_e164(recipient_e164):
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="sms",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="failed",
            error_text="Recipient phone number must be valid E.164.",
        )

    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_SMS_FROM):
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="sms",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="failed",
            error_text="Twilio SMS is not configured.",
        )

    try:
        from twilio.rest import Client

        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body[:1600],
            from_=settings.TWILIO_SMS_FROM,
            to=recipient_e164,
        )
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="sms",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="sent",
            provider_message_id=message.sid,
        )
    except Exception as exc:
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="sms",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="failed",
            error_text=str(exc),
        )


def send_whatsapp_message(
    run_id: str,
    report_request_id: int | None,
    recipient_role: str,
    recipient_e164: str,
    message_type: str,
    body: str,
) -> dict:
    if not is_valid_e164(recipient_e164):
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="whatsapp",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="failed",
            error_text="Recipient phone number must be valid E.164.",
        )

    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_WHATSAPP_FROM):
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="whatsapp",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="failed",
            error_text="Twilio WhatsApp is not configured.",
        )

    try:
        from twilio.rest import Client

        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body[:1600],
            from_=f"whatsapp:{settings.TWILIO_WHATSAPP_FROM}",
            to=f"whatsapp:{recipient_e164}",
        )
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="whatsapp",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="sent",
            provider_message_id=message.sid,
        )
    except Exception as exc:
        return _log_delivery(
            run_id=run_id,
            report_request_id=report_request_id,
            channel="whatsapp",
            recipient_role=recipient_role,
            recipient_e164=recipient_e164,
            message_type=message_type,
            delivery_status="failed",
            error_text=str(exc),
        )


def _log_delivery(
    run_id: str,
    report_request_id: int | None,
    channel: str,
    recipient_role: str,
    recipient_e164: str,
    message_type: str,
    delivery_status: str,
    provider_message_id: str | None = None,
    error_text: str | None = None,
) -> dict:
    row = {
        "run_id": run_id,
        "report_request_id": report_request_id,
        "channel": channel,
        "recipient_role": recipient_role,
        "recipient_e164": recipient_e164,
        "message_type": message_type,
        "delivery_status": delivery_status,
        "provider_message_id": provider_message_id,
        "error_text": error_text,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    inserted = insert_row("message_deliveries", row)
    result = MessageDeliveryResult(
        ok=delivery_status == "sent",
        channel=channel,
        delivery_status=delivery_status,
        provider_message_id=provider_message_id,
        error_text=error_text,
    )
    payload = result.model_dump()
    if inserted:
        payload["id"] = inserted.get("id")
    return payload
