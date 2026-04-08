import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import routes.run as run_module
from main import app


def _install_route_mocks(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(run_module.settings, "UPLOAD_DIR", str(tmp_path / "uploads"))

    def fake_create_run_record(**kwargs):
        captured["create_run_record"] = kwargs

    def fake_create_task(coro):
        captured["task_created"] = True
        coro.close()
        return SimpleNamespace(cancel=lambda: None)

    monkeypatch.setattr(run_module, "create_run_record", fake_create_run_record)
    monkeypatch.setattr(run_module.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(
        run_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="abc123def4567890"),
    )

    return captured


def test_create_run_backwards_compatible(monkeypatch, tmp_path):
    captured = _install_route_mocks(monkeypatch, tmp_path)
    client = TestClient(app)
    fixture = Path(__file__).resolve().parents[1] / "test_data" / "patient_readmission.csv"

    with fixture.open("rb") as handle:
        response = client.post(
            "/api/run",
            files={"file": ("patient_readmission.csv", handle, "text/csv")},
            data={
                "problem_desc": "Predict 30-day hospital readmission risk",
                "target_column": "readmission_30day",
            },
        )

    assert response.status_code == 200
    assert response.json()["run_id"] == "abc123def456"
    assert captured["create_run_record"]["protein_context_json"] is None
    assert captured["task_created"] is True


def test_create_run_accepts_protein_context(monkeypatch, tmp_path):
    captured = _install_route_mocks(monkeypatch, tmp_path)
    client = TestClient(app)
    fixture = Path(__file__).resolve().parents[1] / "test_data" / "patient_readmission_protein_enriched.csv"

    with fixture.open("rb") as handle:
        response = client.post(
            "/api/run",
            files={"file": ("patient_readmission_protein_enriched.csv", handle, "text/csv")},
            data={
                "problem_desc": "Predict 30-day hospital readmission risk",
                "target_column": "readmission_30day",
                "enable_protein_analysis": "true",
                "protein_context_json": json.dumps({
                    "gene_symbol": "TTR",
                    "protein_name": "Transthyretin",
                    "uniprot_id": "P02766",
                    "variant_hgvs": "p.Val142Ile",
                    "disease_label": "Transthyretin Amyloidosis",
                }),
            },
        )

    assert response.status_code == 200
    assert captured["create_run_record"]["protein_context_json"]["gene_symbol"] == "TTR"
    assert captured["create_run_record"]["protein_context_json"]["variant_hgvs"] == "p.Val142Ile"
