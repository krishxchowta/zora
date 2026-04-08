"""
Generate a 500-row multi-disease synthetic dataset for testing Zora's enhanced
30-disease pipeline with realistic clinical value distributions.

Run:
    python test_data/generate_dataset.py

Output:
    test_data/multi_disease_500.csv
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ── Disease list (30) ─────────────────────────────────────────────────────────
DISEASES = [
    "Heart Failure", "COPD", "Diabetes Type 2", "Stroke", "Hypertension",
    "Pneumonia", "Hip Fracture", "Asthma", "UTI", "Gallstones",
    "Kidney Stones", "Appendicitis", "Lung Cancer", "Breast Cancer",
    "Colorectal Cancer", "Prostate Cancer", "Alzheimer's", "Parkinson's",
    "Multiple Sclerosis", "Depression", "Schizophrenia", "Atrial Fibrillation",
    "Coronary Artery Disease", "Deep Vein Thrombosis", "Aortic Aneurysm",
    "Obesity", "Hypothyroidism", "Gout", "Chronic Kidney Disease",
    "COVID-19",
]

N_ROWS = 500
ROWS_PER_DISEASE = N_ROWS // len(DISEASES)   # 16
EXTRA = N_ROWS - ROWS_PER_DISEASE * len(DISEASES)  # fills to 500

# ── Per-disease clinical profile ──────────────────────────────────────────────
# (age_mean, age_std, sbp_mean, glucose_mean, bmi_mean, prior_admissions_mean, readmission_prob)
PROFILES = {
    "Heart Failure":          (72, 8,  155, 110, 28, 3.5, 0.55),
    "COPD":                   (68, 9,  138, 105, 26, 2.8, 0.48),
    "Diabetes Type 2":        (58, 10, 135, 185, 33, 1.8, 0.40),
    "Stroke":                 (67, 11, 162, 115, 27, 2.0, 0.45),
    "Hypertension":           (55, 12, 158, 100, 29, 1.2, 0.28),
    "Pneumonia":              (62, 13, 125, 108, 25, 1.5, 0.38),
    "Hip Fracture":           (78, 7,  132, 100, 24, 2.2, 0.50),
    "Asthma":                 (38, 14, 118, 95,  26, 1.0, 0.22),
    "UTI":                    (55, 15, 120, 98,  27, 0.8, 0.20),
    "Gallstones":             (48, 12, 122, 102, 30, 0.6, 0.18),
    "Kidney Stones":          (45, 11, 128, 98,  27, 0.7, 0.20),
    "Appendicitis":           (32, 12, 115, 95,  25, 0.3, 0.15),
    "Lung Cancer":            (65, 9,  130, 105, 24, 2.5, 0.52),
    "Breast Cancer":          (55, 10, 125, 100, 28, 1.8, 0.38),
    "Colorectal Cancer":      (62, 9,  128, 108, 27, 2.0, 0.42),
    "Prostate Cancer":        (68, 8,  132, 105, 27, 1.5, 0.35),
    "Alzheimer's":            (80, 6,  138, 102, 25, 3.0, 0.60),
    "Parkinson's":            (72, 8,  128, 100, 24, 2.5, 0.50),
    "Multiple Sclerosis":     (40, 10, 118, 95,  25, 1.2, 0.30),
    "Depression":             (42, 13, 120, 100, 28, 1.0, 0.32),
    "Schizophrenia":          (38, 11, 122, 105, 28, 1.5, 0.42),
    "Atrial Fibrillation":    (70, 9,  142, 108, 27, 2.2, 0.45),
    "Coronary Artery Disease":(65, 9,  148, 115, 29, 2.8, 0.50),
    "Deep Vein Thrombosis":   (52, 13, 128, 100, 28, 1.0, 0.28),
    "Aortic Aneurysm":        (68, 9,  145, 108, 27, 1.8, 0.42),
    "Obesity":                (45, 12, 135, 118, 38, 1.2, 0.32),
    "Hypothyroidism":         (50, 12, 120, 98,  30, 0.8, 0.22),
    "Gout":                   (55, 11, 132, 108, 32, 1.0, 0.25),
    "Chronic Kidney Disease": (65, 10, 142, 118, 29, 2.5, 0.52),
    "COVID-19":               (55, 15, 128, 108, 29, 1.5, 0.38),
}

GENDERS = ["M", "F"]
INSURANCE_TYPES = ["private", "medicare", "medicaid", "uninsured"]
ADMISSION_TYPES = ["emergency", "elective", "urgent"]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_rows(disease: str, n: int) -> pd.DataFrame:
    age_m, age_s, sbp_m, gluc_m, bmi_m, prior_m, readm_p = PROFILES[disease]

    age = _clip(rng.normal(age_m, age_s, n), 18, 99).astype(int)
    gender = rng.choice(GENDERS, n)
    systolic_bp = _clip(rng.normal(sbp_m, 14, n), 85, 210).astype(int)
    diastolic_bp = _clip(systolic_bp * rng.uniform(0.55, 0.65, n), 50, 130).astype(int)
    glucose = _clip(rng.normal(gluc_m, 28, n), 55, 420).astype(int)
    bmi = _clip(rng.normal(bmi_m, 4.5, n), 15, 55).round(1)
    prior_admissions = _clip(rng.poisson(prior_m, n), 0, 12).astype(int)
    length_of_stay = _clip(rng.poisson(5 + prior_m, n), 1, 30).astype(int)
    num_medications = _clip(rng.poisson(6 + prior_m * 0.5, n), 1, 20).astype(int)
    num_procedures = _clip(rng.poisson(2 + prior_m * 0.3, n), 0, 10).astype(int)
    creatinine = _clip(rng.normal(1.1 + prior_m * 0.08, 0.35, n), 0.4, 6.0).round(2)
    hemoglobin = _clip(rng.normal(12.5, 1.8, n), 6.0, 18.0).round(1)
    white_blood_cell = _clip(rng.normal(8.5, 2.5, n), 2.0, 25.0).round(1)
    sodium = _clip(rng.normal(139, 3.5, n), 120, 155).astype(int)

    # Cholesterol: ~5% nulls
    cholesterol = rng.normal(190 + bmi * 0.8, 35, n).round(1)
    null_mask_chol = rng.random(n) < 0.05
    cholesterol = cholesterol.astype(object)
    cholesterol[null_mask_chol] = np.nan

    # HbA1c: ~5% nulls, higher for metabolic diseases
    hba1c_base = 5.8 if disease not in ("Diabetes Type 2", "Type 1 Diabetes", "Obesity") else 8.2
    hemoglobin_a1c = _clip(rng.normal(hba1c_base, 1.2, n), 4.0, 14.0).round(1)
    null_mask_hba1c = rng.random(n) < 0.05
    hemoglobin_a1c = hemoglobin_a1c.astype(object)
    hemoglobin_a1c[null_mask_hba1c] = np.nan

    insurance_type = rng.choice(INSURANCE_TYPES, n, p=[0.45, 0.30, 0.15, 0.10])
    admission_type = rng.choice(ADMISSION_TYPES, n, p=[0.50, 0.30, 0.20])

    # Readmission: stochastic with disease-specific base rate
    readmission_30day = (rng.random(n) < readm_p).astype(int)

    return pd.DataFrame({
        "age":               age,
        "gender":            gender,
        "diagnosis":         disease,
        "systolic_bp":       systolic_bp,
        "diastolic_bp":      diastolic_bp,
        "glucose":           glucose,
        "bmi":               bmi,
        "prior_admissions":  prior_admissions,
        "length_of_stay":    length_of_stay,
        "num_medications":   num_medications,
        "num_procedures":    num_procedures,
        "creatinine":        creatinine,
        "hemoglobin":        hemoglobin,
        "white_blood_cell":  white_blood_cell,
        "sodium":            sodium,
        "cholesterol":       cholesterol,
        "hemoglobin_a1c":    hemoglobin_a1c,
        "insurance_type":    insurance_type,
        "admission_type":    admission_type,
        "readmission_30day": readmission_30day,
    })


def main():
    frames = []
    for i, disease in enumerate(DISEASES):
        n = ROWS_PER_DISEASE + (1 if i < EXTRA else 0)
        frames.append(generate_rows(disease, n))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(__file__), "multi_disease_500.csv")
    df.to_csv(out_path, index=False)

    actual_rate = df["readmission_30day"].mean()
    print(f"Written {len(df)} rows × {len(df.columns)} cols → {out_path}")
    print(f"Actual readmission rate: {actual_rate:.1%}")
    print(f"Diseases: {df['diagnosis'].nunique()}")
    print(f"Null cholesterol: {df['cholesterol'].isna().sum()}  "
          f"Null HbA1c: {df['hemoglobin_a1c'].isna().sum()}")


if __name__ == "__main__":
    main()
