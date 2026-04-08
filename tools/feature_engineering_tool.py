import os

import pandas as pd

from models.schemas import FeatureEngineeringReport
from tools.preprocessing_utils import PROTEIN_SIDECAR_COLUMNS, normalize_target_column
from utils.config import settings

AGE_BAND_BINS = [0, 35, 50, 65, 80, 200]
AGE_BAND_LABELS = ["under_35", "35_49", "50_64", "65_79", "80_plus"]
BMI_BAND_BINS = [0, 18.5, 25, 30, 100]
BMI_BAND_LABELS = ["underweight", "healthy", "overweight", "obese"]
LOW_FREQUENCY_THRESHOLD = 3


def feature_engineering_tool(
    run_id: str,
    target_col: str | None = None,
) -> FeatureEngineeringReport:
    filepath = os.path.join(settings.OUTPUT_DIR, run_id, "cleaned.csv")
    df = pd.read_csv(filepath)
    source_rows, source_columns = df.shape

    target_col = normalize_target_column(target_col, df.columns.tolist())
    derived_features_added: list[str] = []
    dropped_from_model_columns: list[str] = []
    rare_category_buckets: dict[str, int] = {}

    parsed_dates: dict[str, pd.Series] = {}
    for col in {"admission_date", "discharge_date"} & set(df.columns):
        parsed_dates[col] = pd.to_datetime(df[col], errors="coerce")

    if "admission_date" in parsed_dates and "discharge_date" in parsed_dates:
        stay_delta = (parsed_dates["discharge_date"] - parsed_dates["admission_date"]).dt.days
        df["length_of_stay_computed"] = stay_delta.clip(lower=0)
        derived_features_added.append("length_of_stay_computed")

    if "admission_date" in parsed_dates:
        df["admission_weekday"] = parsed_dates["admission_date"].dt.day_name().fillna("unknown")
        derived_features_added.append("admission_weekday")

    if "discharge_date" in parsed_dates:
        df["discharge_weekday"] = parsed_dates["discharge_date"].dt.day_name().fillna("unknown")
        derived_features_added.append("discharge_weekday")

    if {"blood_pressure_systolic", "blood_pressure_diastolic"}.issubset(df.columns):
        systolic = pd.to_numeric(df["blood_pressure_systolic"], errors="coerce")
        diastolic = pd.to_numeric(df["blood_pressure_diastolic"], errors="coerce")
        df["pulse_pressure"] = systolic - diastolic
        df["mean_arterial_pressure"] = ((2 * diastolic) + systolic) / 3
        df["bp_high_flag"] = ((systolic >= 140) | (diastolic >= 90)).astype(int)
        derived_features_added.extend(
            ["pulse_pressure", "mean_arterial_pressure", "bp_high_flag"]
        )

    if "age" in df.columns:
        df["age_band"] = pd.cut(
            pd.to_numeric(df["age"], errors="coerce"),
            bins=AGE_BAND_BINS,
            labels=AGE_BAND_LABELS,
            include_lowest=True,
            right=False,
        ).astype("string").fillna("unknown")
        derived_features_added.append("age_band")

    if "bmi" in df.columns:
        df["bmi_band"] = pd.cut(
            pd.to_numeric(df["bmi"], errors="coerce"),
            bins=BMI_BAND_BINS,
            labels=BMI_BAND_LABELS,
            include_lowest=True,
            right=False,
        ).astype("string").fillna("unknown")
        derived_features_added.append("bmi_band")

    if "glucose_level" in df.columns:
        glucose = pd.to_numeric(df["glucose_level"], errors="coerce")
        df["glucose_high_flag"] = (glucose >= 140).astype(int)
        derived_features_added.append("glucose_high_flag")

    if "cholesterol" in df.columns:
        cholesterol = pd.to_numeric(df["cholesterol"], errors="coerce")
        df["cholesterol_high_flag"] = (cholesterol >= 200).astype(int)
        derived_features_added.append("cholesterol_high_flag")

    if "num_prior_admissions" in df.columns:
        prior_admissions = pd.to_numeric(df["num_prior_admissions"], errors="coerce").fillna(0)
        df["prior_admission_flag"] = (prior_admissions > 0).astype(int)
        df["high_utilization_flag"] = (prior_admissions >= 2).astype(int)
        derived_features_added.extend(["prior_admission_flag", "high_utilization_flag"])

    if "smoker" in df.columns:
        smoker = df["smoker"].astype("string").str.lower()
        df["smoker_flag"] = smoker.isin({"yes", "y", "true", "current", "smoker"}).astype(int)
        derived_features_added.append("smoker_flag")

    if "num_medications" in df.columns:
        meds = pd.to_numeric(df["num_medications"], errors="coerce").fillna(0)
        stay_source = None
        if "length_of_stay_days" in df.columns:
            stay_source = pd.to_numeric(df["length_of_stay_days"], errors="coerce")
        elif "length_of_stay_computed" in df.columns:
            stay_source = pd.to_numeric(df["length_of_stay_computed"], errors="coerce")
        if stay_source is not None:
            denominator = stay_source.fillna(0).clip(lower=1)
            df["medication_burden_per_day"] = meds / denominator
            derived_features_added.append("medication_burden_per_day")

    missing_flag_columns = [
        flag for flag in [
            "bmi_missing_flag",
            "cholesterol_missing_flag",
            "glucose_level_missing_flag",
            "blood_pressure_systolic_missing_flag",
            "blood_pressure_diastolic_missing_flag",
        ]
        if flag in df.columns
    ]
    if missing_flag_columns:
        df["lab_missing_burden"] = df[missing_flag_columns].sum(axis=1)
        derived_features_added.append("lab_missing_burden")

    for column in sorted(PROTEIN_SIDECAR_COLUMNS):
        if column in df.columns:
            dropped_from_model_columns.append(column)
    for column in ["admission_date", "discharge_date"]:
        if column in df.columns:
            dropped_from_model_columns.append(column)

    featured_df = df.drop(columns=dropped_from_model_columns, errors="ignore")
    rare_category_buckets = _bucket_low_frequency_categories(featured_df, target_col)

    out_dir = os.path.join(settings.OUTPUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    featured_df.to_csv(os.path.join(out_dir, "featured.csv"), index=False)

    return FeatureEngineeringReport(
        run_id=run_id,
        source_rows=source_rows,
        feature_rows=len(featured_df),
        source_columns=source_columns,
        feature_columns=len(featured_df.columns),
        derived_features_added=derived_features_added,
        dropped_from_model_columns=dropped_from_model_columns,
        rare_category_buckets=rare_category_buckets,
    )


def _bucket_low_frequency_categories(
    df: pd.DataFrame,
    target_col: str | None,
) -> dict[str, int]:
    rare_category_buckets: dict[str, int] = {}
    excluded_cols = {"patient_id", "id", "run_id", target_col}

    for col in df.columns:
        if col in excluded_cols:
            continue
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue

        series = df[col].astype("string")
        counts = series.value_counts(dropna=True)
        rare_values = {
            value for value, count in counts.items()
            if count < LOW_FREQUENCY_THRESHOLD and value not in {"unknown", "other"}
        }
        if not rare_values:
            rare_category_buckets[col] = 0
            continue

        mask = series.isin(list(rare_values))
        df[col] = series.where(~mask, "other")
        rare_category_buckets[col] = int(mask.sum())

    return rare_category_buckets
