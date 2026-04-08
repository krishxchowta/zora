import os

import pandas as pd

from models.schemas import CleanReport, SchemaProfile
from tools.preprocessing_utils import normalize_column_names, normalize_target_column
from utils.config import settings

PLAUSIBILITY_RANGES = {
    "age": (0, 110),
    "blood_pressure_systolic": (60, 260),
    "blood_pressure_diastolic": (30, 150),
    "bmi": (10, 80),
    "glucose_level": (40, 600),
    "cholesterol": (80, 450),
    "length_of_stay_days": (0, 180),
    "num_medications": (0, 50),
    "num_prior_admissions": (0, 50),
}

IMPORTANT_MISSINGNESS_COLUMNS = {
    "bmi",
    "cholesterol",
    "glucose_level",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "length_of_stay_days",
    "num_medications",
    "num_prior_admissions",
}


def clean_tool(
    run_id: str,
    profile: SchemaProfile,
    feedback_ctx: str | None = None,
) -> CleanReport:
    """
    Conservative clinical cleaning pipeline.
    Runs: dedup -> same-visit dedup -> plausibility checks -> missingness flags ->
    impute -> winsorize/cap extremes.

    feedback_ctx is accepted for critic retries and intentionally unused here.
    """
    del feedback_ctx

    filepath = os.path.join(settings.OUTPUT_DIR, run_id, "ingested.csv")
    df = pd.read_csv(filepath)
    df.columns = normalize_column_names(df.columns.tolist())
    rows_before = len(df)

    target_col = normalize_target_column(profile.target_candidate, df.columns.tolist())
    datetime_cols = {
        column for column in df.columns
        if "date" in column or column in {"admission_date", "discharge_date"}
    }

    _normalize_categorical_columns(df)
    _parse_datetime_columns(df, datetime_cols)

    df_before_dedup = len(df)
    df = df.drop_duplicates()
    dupes_removed = df_before_dedup - len(df)

    same_visit_dupes_removed = 0
    if {"patient_id", "admission_date"}.issubset(df.columns):
        before_same_visit = len(df)
        df = df.drop_duplicates(subset=["patient_id", "admission_date"], keep="first")
        same_visit_dupes_removed = before_same_visit - len(df)

    invalid_values_converted = _apply_plausibility_checks(df, target_col)
    missingness_flags_added = _add_missingness_flags(df)

    nulls_imputed: dict[str, int] = {}
    imputation_strategy: dict[str, str] = {}
    numeric_predictors = _numeric_predictor_columns(df, target_col, datetime_cols)

    if target_col and target_col in df.columns:
        target_series = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        df[target_col] = (target_series > 0).astype(int)

    for col in df.columns:
        if col == target_col:
            imputation_strategy[col] = "target_binary_coerce"
            continue

        if col in datetime_cols:
            imputation_strategy[col] = "datetime_preserved"
            continue

        null_count = int(df[col].isna().sum())
        if null_count == 0:
            imputation_strategy[col] = "none"
            continue

        if col in numeric_predictors:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            nulls_imputed[col] = null_count
            imputation_strategy[col] = "median"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            mode_series = df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
            df[col] = df[col].fillna(fill_value)
            nulls_imputed[col] = null_count
            imputation_strategy[col] = "mode" if not mode_series.empty else "constant_unknown"
        else:
            imputation_strategy[col] = "skipped"

    capped_extremes = _cap_extremes(df, numeric_predictors)

    rows_after = len(df)

    out_dir = os.path.join(settings.OUTPUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "cleaned.csv"), index=False)

    return CleanReport(
        run_id=run_id,
        rows_before=rows_before,
        rows_after=rows_after,
        dupes_removed=dupes_removed,
        same_visit_dupes_removed=same_visit_dupes_removed,
        nulls_imputed=nulls_imputed,
        outliers_removed={},
        imputation_strategy=imputation_strategy,
        invalid_values_converted=invalid_values_converted,
        capped_extremes=capped_extremes,
        missingness_flags_added=missingness_flags_added,
    )


def _normalize_categorical_columns(df: pd.DataFrame) -> None:
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            normalized = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
            normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            if col == "smoker":
                normalized = normalized.where(
                    normalized.isna(),
                    normalized.astype(str).str.lower(),
                )
            df[col] = normalized


def _parse_datetime_columns(df: pd.DataFrame, datetime_cols: set[str]) -> None:
    for col in datetime_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")


def _apply_plausibility_checks(df: pd.DataFrame, target_col: str | None) -> dict[str, int]:
    invalid_values_converted: dict[str, int] = {}

    for col, (low, high) in PLAUSIBILITY_RANGES.items():
        if col not in df.columns or col == target_col:
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = numeric_series.notna() & ((numeric_series < low) | (numeric_series > high))
        invalid_values_converted[col] = int(invalid_mask.sum())
        df[f"{col}_invalid_flag"] = invalid_mask.astype(int)
        numeric_series = numeric_series.mask(invalid_mask)
        df[col] = numeric_series

    return invalid_values_converted


def _add_missingness_flags(df: pd.DataFrame) -> list[str]:
    added_flags: list[str] = []
    for col in IMPORTANT_MISSINGNESS_COLUMNS:
        if col not in df.columns:
            continue
        flag_col = f"{col}_missing_flag"
        df[flag_col] = df[col].isna().astype(int)
        added_flags.append(flag_col)
    return added_flags


def _numeric_predictor_columns(
    df: pd.DataFrame,
    target_col: str | None,
    datetime_cols: set[str],
) -> list[str]:
    numeric_cols: list[str] = []
    for col in df.columns:
        if col == target_col or col in datetime_cols:
            continue
        if col.endswith("_flag"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def _cap_extremes(df: pd.DataFrame, numeric_predictors: list[str]) -> dict[str, int]:
    capped_extremes: dict[str, int] = {}

    if len(df) < 20:
        for col in numeric_predictors:
            df[f"{col}_extreme_flag"] = 0
            capped_extremes[col] = 0
        return capped_extremes

    for col in numeric_predictors:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.nunique(dropna=True) <= 1:
            df[f"{col}_extreme_flag"] = 0
            capped_extremes[col] = 0
            continue

        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        if pd.isna(lower) or pd.isna(upper) or lower == upper:
            df[f"{col}_extreme_flag"] = 0
            capped_extremes[col] = 0
            continue

        extreme_mask = series.notna() & ((series < lower) | (series > upper))
        df[f"{col}_extreme_flag"] = extreme_mask.astype(int)
        capped_extremes[col] = int(extreme_mask.sum())
        df[col] = series.clip(lower=lower, upper=upper)

    return capped_extremes
