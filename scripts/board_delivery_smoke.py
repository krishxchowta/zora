#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def api_request(base_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{method} {path} failed with {exc.code}: {body}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive the doctor-board approval flow against a live backend."
    )
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--patient-name", required=True)
    parser.add_argument("--patient-phone")
    parser.add_argument("--patient-whatsapp")
    parser.add_argument(
        "--preferred-channel",
        choices=("sms", "whatsapp"),
        default="sms",
    )
    parser.add_argument(
        "--send-channel",
        choices=("preferred", "sms", "whatsapp"),
        default="sms",
    )
    parser.add_argument("--doctor-name", default="Dr. Demo")
    parser.add_argument(
        "--prescription",
        default="Continue current medications and schedule a 7-day follow-up.",
    )
    parser.add_argument(
        "--request-message",
        default="Please share my approved report and prescription.",
    )
    args = parser.parse_args()

    readiness = api_request(args.base_url, "GET", "/api/ops/readiness")
    print("Readiness summary:")
    print(json.dumps(readiness, indent=2))
    if not readiness.get("database_ready"):
        print(
            "\nDatabase is not ready. Apply the migration bundle before running the smoke flow.",
            file=sys.stderr,
        )
        return 2
    if args.send_channel == "sms" and not args.patient_phone:
        print("--patient-phone is required for SMS validation.", file=sys.stderr)
        return 2
    if args.send_channel == "whatsapp" and not args.patient_whatsapp:
        print("--patient-whatsapp is required for WhatsApp validation.", file=sys.stderr)
        return 2

    request_payload = {
        "run_id": args.run_id,
        "patient_name": args.patient_name,
        "phone_e164": args.patient_phone,
        "whatsapp_e164": args.patient_whatsapp,
        "preferred_channel": args.preferred_channel,
        "request_message": args.request_message,
    }
    report_request = api_request(
        args.base_url,
        "POST",
        "/api/report-requests",
        request_payload,
    )
    report_request_id = report_request["id"]
    print("\nCreated report request:")
    print(json.dumps(report_request, indent=2))

    encoded_id = urllib.parse.urlencode({"report_request_id": report_request_id})
    prescription = api_request(
        args.base_url,
        "POST",
        f"/api/board/cases/{args.run_id}/prescription?{encoded_id}",
        {
            "doctor_name": args.doctor_name,
            "prescription_text": args.prescription,
            "notes": "Saved by smoke-test runner.",
        },
    )
    print("\nSaved prescription draft:")
    print(json.dumps(prescription, indent=2))

    approval = api_request(
        args.base_url,
        "POST",
        f"/api/report-requests/{report_request_id}/approve",
        {
            "doctor_name": args.doctor_name,
            "prescription_text": args.prescription,
            "notes": "Approved by smoke-test runner.",
            "send_channel": args.send_channel,
        },
    )
    print("\nApproval result:")
    print(json.dumps(approval, indent=2))

    delivery = approval.get("delivery", {})
    if not delivery.get("ok"):
        print(
            "\nDelivery did not succeed. Check readiness, Twilio config, and delivery logs.",
            file=sys.stderr,
        )
        return 1

    print("\nSmoke flow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
