"""
Safety Vault — deterministic, LLM-free medical rule engine.

Hard-coded JSON rules checked against the ML output and AlphaFold result.
Any triggered rule:
  - Sets doctor_review = True
  - Adds a flag entry to safety_flags list
  - Optionally substitutes a safe_value for the affected field

This runs BEFORE the synthesis agent so downstream narration
reflects safety overrides.
"""


def _check_rules(
    ml_auc: float,
    ml_accuracy: float,
    stability_score: float,
    denial_probability: float,
    waste_estimate_usd: float,
    protein_name: str,
    misfold_summary: dict | None = None,
) -> list[dict]:
    """
    Evaluate all medical safety rules with explicit comparisons.
    Returns list of triggered rule dicts.
    """
    triggered: list[dict] = []

    if ml_auc >= 0.85:
        triggered.append({
            "rule_id": "SR-001",
            "name":    "High readmission risk threshold",
            "action":  "doctor_review",
            "message": "Model AUC ≥ 0.85 — high-confidence readmission risk. Mandatory physician review.",
        })

    if stability_score < 0.40:
        triggered.append({
            "rule_id": "SR-002",
            "name":    "Unstable protein — chronic severity",
            "action":  "doctor_review",
            "message": "Protein stability < 0.40 — structurally compromised biomarker. Escalate to specialist.",
        })

    if denial_probability >= 0.55:
        triggered.append({
            "rule_id": "SR-003",
            "name":    "High insurance denial risk",
            "action":  "flag",
            "message": "Denial probability ≥ 55%. Recommend pre-authorization and case management.",
        })

    if waste_estimate_usd >= 500_000:
        triggered.append({
            "rule_id": "SR-004",
            "name":    "Extreme waste estimate",
            "action":  "flag",
            "message": "Projected healthcare waste ≥ $500K. Recommend population health intervention.",
        })

    if protein_name in ("BNP", "PLAT"):
        triggered.append({
            "rule_id": "SR-005",
            "name":    "High-risk cardiac/stroke biomarker",
            "action":  "doctor_review",
            "message": "Cardiac/stroke biomarker detected. All predictions require cardiologist sign-off.",
        })

    if ml_accuracy < 0.60:
        triggered.append({
            "rule_id":   "SR-006",
            "name":      "Low model accuracy — uncertain prediction",
            "action":    "override",
            "safe_field":"ml_auc",
            "safe_value": None,
            "message":   "Model accuracy < 60%. Prediction unreliable. Suppressing AUC from patient output.",
        })

    if misfold_summary and misfold_summary.get("enabled"):
        evidence = misfold_summary.get("evidence", [])
        cpad_match = next(
            (
                item for item in evidence
                if item.get("source") == "CPAD Fixture"
                and item.get("type") == "exact_variant_match"
                and float(item.get("aggregation_increase_pct", 0)) > 50
            ),
            None,
        )
        if cpad_match:
            triggered.append({
                "rule_id": "SR-007",
                "name": "High aggregation mutation match",
                "action": "doctor_review",
                "message": (
                    f"CPAD-derived evidence for {cpad_match.get('variant_hgvs')} shows "
                    f"{cpad_match.get('aggregation_increase_pct')}% higher aggregation. "
                    "Mandatory protein-misfold review."
                ),
            })

        energy_state = misfold_summary.get("energy_state")
        stuck_score = float(misfold_summary.get("stuck_score") or 0)
        if stuck_score >= 0.75 or energy_state in ("toxic_intermediate", "aggregation_prone"):
            triggered.append({
                "rule_id": "SR-008",
                "name": "Aggregation-prone energy state",
                "action": "doctor_review",
                "message": (
                    f"Misfold stage indicates {energy_state or 'high-risk'} behavior "
                    f"(stuck-score={misfold_summary.get('stuck_score')}). "
                    "Escalate to specialist review."
                ),
            })

        hotspot_regions = misfold_summary.get("viewer_stub", {}).get("hotspot_regions", [])
        surface_exposure = misfold_summary.get("surface_exposure_score")
        if surface_exposure is not None and float(surface_exposure) >= 0.65 and hotspot_regions:
            triggered.append({
                "rule_id": "SR-009",
                "name": "Surface hotspot exposure",
                "action": "flag",
                "message": (
                    f"Surface hydrophobic exposure is elevated ({surface_exposure}) in hotspot "
                    f"regions {', '.join(hotspot_regions[:3])}. Monitor aggregation risk."
                ),
            })

        if misfold_summary.get("variant_hgvs") and misfold_summary.get("variant_delta_score") is None:
            triggered.append({
                "rule_id": "SR-010",
                "name": "Variant lacks curated evidence",
                "action": "flag",
                "message": (
                    f"Variant {misfold_summary.get('variant_hgvs')} has no curated CPAD-derived "
                    "delta score. Treat the protein interpretation as research-only."
                ),
            })

    return triggered


def run_safety_vault(
    ml_auc: float,
    ml_accuracy: float,
    stability_score: float,
    denial_probability: float,
    waste_estimate_usd: float,
    protein_name: str,
    misfold_summary: dict | None = None,
) -> dict:
    """
    Evaluate all safety rules against pipeline outputs.

    Returns:
        doctor_review: bool
        safety_flags:  list[dict]  — triggered rules with messages
        overrides:     dict        — field → safe_value substitutions
    """
    triggered = _check_rules(
        ml_auc, ml_accuracy, stability_score,
        denial_probability, waste_estimate_usd, protein_name,
        misfold_summary=misfold_summary,
    )

    doctor_review = any(r["action"] == "doctor_review" for r in triggered)
    overrides = {
        r["safe_field"]: r.get("safe_value")
        for r in triggered
        if r["action"] == "override" and "safe_field" in r
    }

    return {
        "doctor_review":    doctor_review,
        "safety_flags":     triggered,
        "overrides":        overrides,
        "rules_checked":    10 if misfold_summary and misfold_summary.get("enabled") else 6,
        "rules_triggered":  len(triggered),
    }
