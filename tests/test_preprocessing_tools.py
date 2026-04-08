from pathlib import Path

import pandas as pd

from models.schemas import SchemaProfile
from tools.automl_tool import _get_modeling_input_path
from tools.clean_tool import clean_tool
from tools.feature_engineering_tool import feature_engineering_tool
from utils.config import settings


def _seed_ingested_csv(tmp_path: Path, run_id: str, frame: pd.DataFrame) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "ingested.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def _profile_for_frame(frame: pd.DataFrame, target_candidate: str = "readmission_30day") -> SchemaProfile:
    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [
        column for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column])
    ]
    return SchemaProfile(
        run_id="test-run",
        filename="fixture.csv",
        rows=len(frame),
        cols=len(frame.columns),
        columns=[],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=[],
        target_candidate=target_candidate,
        null_summary={},
        duplicate_count=0,
        memory_mb=0.01,
    )


def test_clean_tool_converts_invalid_values_and_adds_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(tmp_path))
    run_id = "clean-invalids"
    frame = pd.DataFrame(
        [
            {
                "patient_id": "A1",
                "admission_date": "2026-01-01",
                "age": 140,
                "cholesterol": None,
                "glucose_level": 120,
                "smoker": "Yes",
                "readmission_30day": 1,
            },
            {
                "patient_id": "A1",
                "admission_date": "2026-01-01",
                "age": 150,
                "cholesterol": None,
                "glucose_level": 121,
                "smoker": "Yes",
                "readmission_30day": 1,
            },
            {
                "patient_id": "A2",
                "admission_date": "2026-01-03",
                "age": 60,
                "cholesterol": 1000,
                "glucose_level": 130,
                "smoker": "No",
                "readmission_30day": 0,
            },
            {
                "patient_id": "A3",
                "admission_date": "2026-01-04",
                "age": 58,
                "cholesterol": 210,
                "glucose_level": 145,
                "smoker": "No",
                "readmission_30day": 1,
            },
        ]
    )
    _seed_ingested_csv(tmp_path, run_id, frame)

    report = clean_tool(run_id=run_id, profile=_profile_for_frame(frame))
    cleaned = pd.read_csv(tmp_path / run_id / "cleaned.csv")

    assert report.rows_before == 4
    assert report.rows_after == 3
    assert report.same_visit_dupes_removed == 1
    assert report.invalid_values_converted["age"] == 1
    assert report.invalid_values_converted["cholesterol"] == 1
    assert report.nulls_imputed["cholesterol"] == 2
    assert "cholesterol_missing_flag" in report.missingness_flags_added
    assert cleaned["age_invalid_flag"].sum() == 1
    assert cleaned["cholesterol_invalid_flag"].sum() == 1
    assert cleaned["cholesterol_missing_flag"].sum() == 2
    assert cleaned["readmission_30day"].isin([0, 1]).all()


def test_clean_tool_caps_extremes_without_row_drops(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(tmp_path))
    run_id = "clean-capping"
    rows = []
    for index in range(25):
        rows.append(
            {
                "patient_id": f"P{index}",
                "age": 40 + index,
                "cholesterol": 180 + index,
                "glucose_level": 100 + index,
                "readmission_30day": index % 2,
            }
        )
    rows[-1]["glucose_level"] = 590
    frame = pd.DataFrame(rows)
    _seed_ingested_csv(tmp_path, run_id, frame)

    report = clean_tool(run_id=run_id, profile=_profile_for_frame(frame))
    cleaned = pd.read_csv(tmp_path / run_id / "cleaned.csv")

    assert report.rows_after == 25
    assert report.capped_extremes["glucose_level"] >= 1
    assert "glucose_level_extreme_flag" in cleaned.columns
    assert cleaned["glucose_level_extreme_flag"].sum() >= 1


def test_feature_engineering_tool_builds_interpretable_feature_matrix(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(tmp_path))
    run_id = "feature-run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cleaned = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "admission_date": "2026-01-01",
                "discharge_date": "2026-01-05",
                "blood_pressure_systolic": 150,
                "blood_pressure_diastolic": 92,
                "age": 67,
                "bmi": 31.2,
                "glucose_level": 148,
                "cholesterol": 220,
                "num_medications": 10,
                "length_of_stay_days": 4,
                "num_prior_admissions": 2,
                "smoker": "yes",
                "department": "cardiology",
                "cholesterol_missing_flag": 0,
                "glucose_level_missing_flag": 0,
                "blood_pressure_systolic_missing_flag": 0,
                "blood_pressure_diastolic_missing_flag": 0,
                "gene_symbol": "TTR",
                "protein_name": "TTR",
                "uniprot_id": "P02766",
                "variant_hgvs": "p.Val142Ile",
                "surface_hydrophobic_ratio": 0.77,
                "critical_region_id": "TTR-beta-sheet-F",
                "disease_label": "Amyloidosis",
                "readmission_30day": 1,
            },
            {
                "patient_id": "P2",
                "admission_date": "2026-01-02",
                "discharge_date": "2026-01-04",
                "blood_pressure_systolic": 132,
                "blood_pressure_diastolic": 84,
                "age": 52,
                "bmi": 27.1,
                "glucose_level": 138,
                "cholesterol": 198,
                "num_medications": 7,
                "length_of_stay_days": 2,
                "num_prior_admissions": 1,
                "smoker": "no",
                "department": "cardiology",
                "cholesterol_missing_flag": 1,
                "glucose_level_missing_flag": 0,
                "blood_pressure_systolic_missing_flag": 0,
                "blood_pressure_diastolic_missing_flag": 0,
                "gene_symbol": "TTR",
                "protein_name": "TTR",
                "uniprot_id": "P02766",
                "variant_hgvs": "p.Val142Ile",
                "surface_hydrophobic_ratio": 0.77,
                "critical_region_id": "TTR-beta-sheet-F",
                "disease_label": "Amyloidosis",
                "readmission_30day": 0,
            },
            {
                "patient_id": "P3",
                "admission_date": "2026-01-03",
                "discharge_date": "2026-01-05",
                "blood_pressure_systolic": 128,
                "blood_pressure_diastolic": 78,
                "age": 41,
                "bmi": 24.0,
                "glucose_level": 109,
                "cholesterol": 175,
                "num_medications": 5,
                "length_of_stay_days": 2,
                "num_prior_admissions": 0,
                "smoker": "no",
                "department": "oncology",
                "cholesterol_missing_flag": 0,
                "glucose_level_missing_flag": 1,
                "blood_pressure_systolic_missing_flag": 0,
                "blood_pressure_diastolic_missing_flag": 0,
                "gene_symbol": "TP53",
                "protein_name": "TP53",
                "uniprot_id": "P04637",
                "variant_hgvs": "p.Mock3",
                "surface_hydrophobic_ratio": 0.42,
                "critical_region_id": "TP53-core",
                "disease_label": "Cancer",
                "readmission_30day": 1,
            },
            {
                "patient_id": "P4",
                "admission_date": "2026-01-04",
                "discharge_date": "2026-01-06",
                "blood_pressure_systolic": 124,
                "blood_pressure_diastolic": 76,
                "age": 38,
                "bmi": 22.5,
                "glucose_level": 102,
                "cholesterol": 168,
                "num_medications": 4,
                "length_of_stay_days": 2,
                "num_prior_admissions": 0,
                "smoker": "no",
                "department": "rare_unit",
                "cholesterol_missing_flag": 0,
                "glucose_level_missing_flag": 0,
                "blood_pressure_systolic_missing_flag": 0,
                "blood_pressure_diastolic_missing_flag": 1,
                "gene_symbol": "TP53",
                "protein_name": "TP53",
                "uniprot_id": "P04637",
                "variant_hgvs": "p.Mock4",
                "surface_hydrophobic_ratio": 0.41,
                "critical_region_id": "TP53-core",
                "disease_label": "Cancer",
                "readmission_30day": 0,
            },
        ]
    )
    cleaned.to_csv(run_dir / "cleaned.csv", index=False)

    report = feature_engineering_tool(run_id=run_id, target_col="readmission_30day")
    featured = pd.read_csv(run_dir / "featured.csv")

    assert "length_of_stay_computed" in featured.columns
    assert "pulse_pressure" in featured.columns
    assert "mean_arterial_pressure" in featured.columns
    assert "bp_high_flag" in featured.columns
    assert "age_band" in featured.columns
    assert "bmi_band" in featured.columns
    assert "lab_missing_burden" in featured.columns
    assert "smoker_flag" in featured.columns
    assert "gene_symbol" not in featured.columns
    assert "protein_name" not in featured.columns
    assert "admission_date" not in featured.columns
    assert "discharge_date" not in featured.columns
    assert report.rare_category_buckets["department"] == 4
    assert "department" in featured.columns
    assert (featured["department"] == "other").sum() == 4


def test_automl_prefers_featured_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(tmp_path))
    run_id = "automl-path"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cleaned.csv").write_text("a\n1\n")
    (run_dir / "featured.csv").write_text("a\n2\n")

    assert _get_modeling_input_path(run_id).endswith("featured.csv")
