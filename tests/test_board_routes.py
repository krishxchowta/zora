from fastapi.testclient import TestClient

from models.schemas import BoardCaseDetail, BoardCaseSummary, OpsReadinessReport
import routes.board as board_module
import routes.ops as ops_module
from main import app


def test_get_board_cases(monkeypatch):
    monkeypatch.setattr(
        board_module,
        "list_board_cases",
        lambda: [
            BoardCaseSummary(
                run_id="run123",
                filename="patient_readmission.csv",
                pipeline_status="full_complete",
                request_status="doctor_notified",
                message_status="sent",
                patient_name="Asha",
                doctor_review=True,
                doctor_report_ready=True,
                patient_report_ready=True,
            )
        ],
    )
    client = TestClient(app)

    response = client.get("/api/board/cases")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cases"][0]["run_id"] == "run123"
    assert payload["cases"][0]["doctor_review"] is True


def test_get_board_case_detail(monkeypatch):
    monkeypatch.setattr(
        board_module,
        "get_board_case_detail",
        lambda run_id: BoardCaseDetail(
            run={"run_id": run_id, "status": "full_complete"},
            insight={"doctor_review": True},
            patient_contacts=[],
            report_requests=[],
            prescriptions=[],
            message_deliveries=[],
            doctor_report_text="Doctor draft",
            patient_report_text="Patient draft",
            final_prescription_text=None,
        ),
    )
    client = TestClient(app)

    response = client.get("/api/board/cases/run789")

    assert response.status_code == 200
    assert response.json()["doctor_report_text"] == "Doctor draft"


def test_create_report_request(monkeypatch):
    monkeypatch.setattr(
        board_module,
        "create_patient_report_request",
        lambda payload: {
            "id": 17,
            "run_id": payload["run_id"],
            "status": "doctor_notified",
            "patient_contact_id": 9,
        },
    )
    client = TestClient(app)

    response = client.post(
        "/api/report-requests",
        json={
            "run_id": "run555",
            "patient_name": "Ravi",
            "phone_e164": "+919876543210",
            "preferred_channel": "sms",
            "request_message": "Please share my report.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "doctor_notified"


def test_approve_report_request(monkeypatch):
    monkeypatch.setattr(
        board_module,
        "approve_report_request",
        lambda report_request_id, doctor_name, prescription_text, notes, send_channel: {
            "report_request_id": report_request_id,
            "status": "sent",
            "doctor_name": doctor_name,
            "prescription_text": prescription_text,
            "send_channel": send_channel,
        },
    )
    client = TestClient(app)

    response = client.post(
        "/api/report-requests/42/approve",
        json={
            "doctor_name": "Dr. Mehta",
            "prescription_text": "Start follow-up in 7 days.",
            "notes": "Call patient before discharge.",
            "send_channel": "whatsapp",
        },
    )

    assert response.status_code == 200
    assert response.json()["report_request_id"] == 42
    assert response.json()["send_channel"] == "whatsapp"


def test_reject_report_request_requires_reason():
    client = TestClient(app)

    response = client.post(
        "/api/report-requests/42/reject",
        json={
            "doctor_name": "Dr. Mehta",
            "reason": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "A rejection reason is required."


def test_send_whatsapp_route(monkeypatch):
    monkeypatch.setattr(
        board_module,
        "send_report_request_message",
        lambda report_request_id, channel, doctor_name, notes: {
            "report_request_id": report_request_id,
            "delivery": {"ok": True, "channel": channel},
            "doctor_name": doctor_name,
        },
    )
    client = TestClient(app)

    response = client.post(
        "/api/report-requests/55/send-whatsapp",
        json={
            "doctor_name": "Dr. Rao",
            "notes": "Patient requested copy.",
        },
    )

    assert response.status_code == 200
    assert response.json()["delivery"]["channel"] == "whatsapp"


def test_get_ops_readiness(monkeypatch):
    monkeypatch.setattr(
        ops_module,
        "get_readiness_report",
        lambda: OpsReadinessReport(
            overall_ready=False,
            database_ready=False,
            sms_ready=True,
            whatsapp_ready=False,
            doctor_notification_ready=False,
            board_delivery_ready=False,
            migration_bundle_path="/tmp/bundle.sql",
            checks=[],
            required_manual_steps=["Apply SQL"],
        ),
    )
    client = TestClient(app)

    response = client.get("/api/ops/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["database_ready"] is False
    assert payload["migration_bundle_path"] == "/tmp/bundle.sql"
