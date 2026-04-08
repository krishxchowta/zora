import os
import warnings
import pandas as pd
import numpy as np
from tools.preprocessing_utils import (
    PROTEIN_SIDECAR_COLUMNS,
    normalize_target_column,
)
from utils.config import settings

warnings.filterwarnings("ignore")


# Diagnosis → protein mapping for AlphaFold integration (30 diseases)
DIAGNOSIS_PROTEIN_MAP = {
    # Original 12
    "Heart Failure":         ("BNP",      "P16860"),
    "COPD":                  ("SERPINA1", "P01009"),
    "Diabetes Type 2":       ("GCK",      "P35557"),
    "Stroke":                ("PLAT",     "P00750"),
    "Hypertension":          ("ACE",      "P12821"),
    "Pneumonia":             ("DEFB1",    "P60022"),
    "Hip Fracture":          ("RANKL",    "O14788"),
    "Asthma":                ("IL13",     "P35225"),
    "UTI":                   ("TLR4",     "O00206"),
    "Gallstones":            ("ABCG8",    "Q9Y210"),
    "Kidney Stones":         ("SLC34A1",  "Q06495"),
    "Appendicitis":          ("CRP",      "P02741"),
    # Cancer
    "Lung Cancer":           ("EGFR",     "P00533"),
    "Breast Cancer":         ("BRCA1",    "P38398"),
    "Colorectal Cancer":     ("APC",      "P25054"),
    "Prostate Cancer":       ("KLK3",     "P07288"),
    # Neurological
    "Alzheimer's":           ("APP",      "P05067"),
    "Parkinson's":           ("SNCA",     "P37840"),
    "Multiple Sclerosis":    ("MBP",      "P02686"),
    "Depression":            ("SLC6A4",   "P31645"),
    "Schizophrenia":         ("DTNBP1",   "Q9Y228"),
    # Cardiovascular / Metabolic
    "Atrial Fibrillation":   ("SCN5A",    "Q14524"),
    "Coronary Artery Disease":("PCSK9",   "Q8NBP7"),
    "Deep Vein Thrombosis":  ("F2",       "P00734"),
    "Aortic Aneurysm":       ("FBN1",     "P35555"),
    "Obesity":               ("LEP",      "P41159"),
    "Hypothyroidism":        ("TSHR",     "P16473"),
    "Gout":                  ("HPRT1",    "P00492"),
    # Organ / Immune
    "Chronic Kidney Disease":("UMOD",     "Q9UI40"),
    "Crohn's Disease":       ("NOD2",     "Q9HC29"),
    "Pancreatitis":          ("PRSS1",    "P07477"),
    "Liver Cirrhosis":       ("ALB",      "P02768"),
    "Rheumatoid Arthritis":  ("TNF",      "P01375"),
    "Lupus":                 ("TREX1",    "Q9NSU2"),
    "Type 1 Diabetes":       ("INS",      "P01308"),
    "Sepsis":                ("LBP",      "P18428"),
    "Tuberculosis":          ("MPT64",    "P9WMX5"),
    "COVID-19":              ("ACE2",     "Q9BYF1"),
}
DEFAULT_PROTEIN = ("TP53", "P04637")


def automl_tool(run_id: str, target_col: str) -> dict:
    """
    Run PyCaret classification pipeline on featured.csv (or cleaned.csv as fallback).

    Enhancements vs v1:
    - 9 model families for datasets ≥50 rows; 5 for small
    - Dynamic k-fold: 5 if ≥100 rows, else 3
    - Automatic class-imbalance detection + SMOTE fix
    - Multicollinearity removal (threshold 0.9)
    - tune_model (AUC, n_iter=10) for ≥50 rows
    - calibrate_model for better probability estimates
    - save_model → outputs/{run_id}/best_model.pkl
    - Returns 7 new metadata fields (backward-compatible)
    """
    filepath = _get_modeling_input_path(run_id)
    df = pd.read_csv(filepath)
    target_col = normalize_target_column(target_col, df.columns.tolist()) or target_col

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in cleaned data.")

    # Drop non-predictive ID columns and protein sidecar columns
    id_cols = [c for c in df.columns if c.lower() in ("patient_id", "id", "run_id")]
    df = df.drop(columns=id_cols + sorted(PROTEIN_SIDECAR_COLUMNS), errors="ignore")

    # Ensure target is binary int
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    # Drop date-like columns PyCaret can't handle
    date_cols = [c for c in df.columns if "date" in c.lower()]
    df = df.drop(columns=date_cols, errors="ignore")

    n_rows = len(df)

    # Dynamic fold count based on dataset size
    fold = 5 if n_rows >= 100 else 3

    # Class imbalance detection
    class_counts = df[target_col].value_counts()
    if len(class_counts) >= 2:
        imbalance_ratio = float(class_counts.iloc[0]) / float(class_counts.iloc[-1])
    else:
        imbalance_ratio = 1.0
    fix_imbalance = imbalance_ratio > 2.0

    # Model list depends on dataset size
    if n_rows >= 50:
        model_list = ["lr", "dt", "rf", "lightgbm", "xgboost", "et", "gbc", "ada", "nb"]
    else:
        model_list = ["lr", "dt", "rf", "lightgbm", "nb"]

    from pycaret.classification import (
        setup as pc_setup,
        compare_models,
        tune_model,
        calibrate_model,
        pull,
        save_model,
    )

    pc_setup(
        data=df,
        target=target_col,
        session_id=42,
        verbose=False,
        html=False,
        n_jobs=1,
        fold=fold,
        fix_imbalance=fix_imbalance,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.9,
    )

    best = compare_models(
        include=model_list,
        n_select=1,
        sort="AUC",
        verbose=False,
        turbo=True,
    )

    # Tune model for larger datasets
    tuning_applied = False
    if n_rows >= 50:
        try:
            best = tune_model(best, optimize="AUC", n_iter=10, verbose=False)
            tuning_applied = True
        except Exception:
            pass  # keep untuned best

    # Calibrate probabilities
    calibration_applied = False
    try:
        best = calibrate_model(best, verbose=False)
        calibration_applied = True
    except Exception:
        pass

    # Pull leaderboard after all steps
    results_df = pull()
    best_row = results_df.iloc[0]
    model_name = type(best).__name__

    metrics = {
        "model":     model_name,
        "accuracy":  round(float(best_row.get("Accuracy",  0)), 4),
        "auc":       round(float(best_row.get("AUC",       0)), 4),
        "f1":        round(float(best_row.get("F1",        0)), 4),
        "recall":    round(float(best_row.get("Recall",    0)), 4),
        # Guard against PyCaret version column-name drift
        "precision": round(float(best_row.get("Prec.", best_row.get("Precision", 0))), 4),
    }

    # Save model artifact
    model_saved_path = ""
    try:
        out_dir = os.path.join(settings.OUTPUT_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)
        artifact_path = os.path.join(out_dir, "best_model")
        save_model(best, artifact_path)
        model_saved_path = artifact_path + ".pkl"
    except Exception:
        pass

    # SHAP feature importance
    feature_importance = _compute_shap(best, df.drop(columns=[target_col]))

    return {
        # Original 4 keys — unchanged
        "model_name":   model_name,
        "metrics":      metrics,
        "top_features": feature_importance,
        "source_file":  os.path.basename(filepath),
        # New metadata fields
        "fold_count":              fold,
        "imbalance_ratio":         round(imbalance_ratio, 3),
        "fix_imbalance_applied":   fix_imbalance,
        "models_evaluated":        model_list,
        "tuning_applied":          tuning_applied,
        "calibration_applied":     calibration_applied,
        "model_saved_path":        model_saved_path,
    }


def _compute_shap(model, X: pd.DataFrame) -> dict:
    """Return top-10 SHAP feature importances as {feature: mean_abs_shap}."""
    try:
        import shap
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include="object").columns:
            X_enc[col] = pd.Categorical(X_enc[col]).codes

        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_enc, min(20, len(X_enc)))
            )

        shap_values = explainer.shap_values(X_enc)
        # Binary classification: shap_values may be [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(X_enc.columns.tolist(), mean_abs.tolist()))
        top10 = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        return {k: round(v, 5) for k, v in top10.items()}
    except Exception:
        return {col: round(1.0 / len(X.columns), 5) for col in X.columns[:10]}


def get_protein_for_run(run_id: str) -> tuple[str, str]:
    """
    Read cleaned.csv, find the most frequent diagnosis, return (protein_name, uniprot_id).
    """
    filepath = os.path.join(settings.OUTPUT_DIR, run_id, "cleaned.csv")
    try:
        df = pd.read_csv(filepath)
        if "diagnosis" in df.columns:
            top_diagnosis = df["diagnosis"].mode()[0]
            return DIAGNOSIS_PROTEIN_MAP.get(top_diagnosis, DEFAULT_PROTEIN)
    except Exception:
        pass
    return DEFAULT_PROTEIN


def _get_modeling_input_path(run_id: str) -> str:
    featured_path = os.path.join(settings.OUTPUT_DIR, run_id, "featured.csv")
    if os.path.exists(featured_path):
        return featured_path
    return os.path.join(settings.OUTPUT_DIR, run_id, "cleaned.csv")
