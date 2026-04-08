from pathlib import Path
import shutil

from models.schemas import ProteinContext
from tools.misfold_tool import misfold_tool, resolve_protein_context_for_run
from tools.safety_vault import run_safety_vault
from utils.config import settings


def _seed_run(monkeypatch, tmp_path: Path, fixture_name: str = "patient_readmission_protein_enriched.csv") -> str:
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(tmp_path))
    run_id = "protein-run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).resolve().parents[1] / "test_data" / fixture_name
    shutil.copyfile(fixture, run_dir / "cleaned.csv")
    return run_id


def test_protein_context_precedence(monkeypatch, tmp_path):
    run_id = _seed_run(monkeypatch, tmp_path)

    dataset_context = resolve_protein_context_for_run(run_id)
    assert dataset_context.gene_symbol == "TTR"
    assert dataset_context.protein_name == "TTR"
    assert dataset_context.uniprot_id == "P02766"

    explicit_context = ProteinContext(
        gene_symbol="NPPB",
        protein_name="BNP",
        uniprot_id="P16860",
        variant_hgvs="p.Mock1",
    )
    resolved = resolve_protein_context_for_run(run_id, explicit_context)
    assert resolved.gene_symbol == "NPPB"
    assert resolved.protein_name == "BNP"
    assert resolved.uniprot_id == "P16860"
    assert resolved.variant_hgvs == "p.Mock1"


def test_misfold_scoring_and_flags(monkeypatch, tmp_path):
    run_id = _seed_run(monkeypatch, tmp_path)
    context = resolve_protein_context_for_run(run_id)

    summary = misfold_tool(
        run_id=run_id,
        protein_context=context,
        alphafold_result={
            "protein_name": context.protein_name,
            "pdb_link": f"https://alphafold.ebi.ac.uk/entry/{context.uniprot_id}",
        },
    )

    assert summary.enabled is True
    assert summary.protein_name == "TTR"
    assert summary.variant_hgvs == "p.Val142Ile"
    assert summary.aggregation_propensity == 0.83
    assert summary.variant_delta_score == 0.84
    assert summary.energy_state == "toxic_intermediate"
    assert summary.viewer_stub["render_status"] == "placeholder"
    assert "TTR-beta-sheet-F" in summary.viewer_stub["hotspot_regions"]

    safety = run_safety_vault(
        ml_auc=0.91,
        ml_accuracy=0.89,
        stability_score=0.78,
        denial_probability=0.31,
        waste_estimate_usd=44311,
        protein_name=summary.protein_name or "TTR",
        misfold_summary=summary.model_dump(),
    )
    rule_ids = {item["rule_id"] for item in safety["safety_flags"]}
    assert {"SR-007", "SR-008", "SR-009"}.issubset(rule_ids)
    assert safety["doctor_review"] is True


def test_misfold_null_variant_evidence(monkeypatch, tmp_path):
    run_id = _seed_run(monkeypatch, tmp_path)
    explicit_context = ProteinContext(variant_hgvs="p.UnknownVariant")
    context = resolve_protein_context_for_run(run_id, explicit_context)

    summary = misfold_tool(
        run_id=run_id,
        protein_context=context,
        alphafold_result={"pdb_link": f"https://alphafold.ebi.ac.uk/entry/{context.uniprot_id}"},
    )

    assert summary.variant_hgvs == "p.UnknownVariant"
    assert summary.variant_delta_score is None
    assert any(item["type"] == "no_curated_match" for item in summary.evidence)

    safety = run_safety_vault(
        ml_auc=0.7,
        ml_accuracy=0.82,
        stability_score=0.74,
        denial_probability=0.2,
        waste_estimate_usd=1000,
        protein_name=summary.protein_name or "TTR",
        misfold_summary=summary.model_dump(),
    )
    rule_ids = {item["rule_id"] for item in safety["safety_flags"]}
    assert "SR-010" in rule_ids
