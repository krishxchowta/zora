from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
import json
import os

import pandas as pd

from models.schemas import MisfoldSummary, ProteinContext
from utils.config import settings

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "reference_data" / "misfold"
DATASET_CONTEXT_COLUMNS = (
    "gene_symbol",
    "protein_name",
    "uniprot_id",
    "variant_hgvs",
    "disease_label",
    "notes",
)

DIAGNOSIS_PROTEIN_MAP = {
    "Heart Failure": {
        "gene_symbol": "NPPB",
        "protein_name": "BNP",
        "uniprot_id": "P16860",
        "disease_label": "Heart Failure",
    },
    "COPD": {
        "gene_symbol": "SERPINA1",
        "protein_name": "SERPINA1",
        "uniprot_id": "P01009",
        "disease_label": "COPD",
    },
    "Diabetes Type 2": {
        "gene_symbol": "GCK",
        "protein_name": "GCK",
        "uniprot_id": "P35557",
        "disease_label": "Diabetes Type 2",
    },
    "Stroke": {
        "gene_symbol": "PLAT",
        "protein_name": "PLAT",
        "uniprot_id": "P00750",
        "disease_label": "Stroke",
    },
    "Hypertension": {
        "gene_symbol": "ACE",
        "protein_name": "ACE",
        "uniprot_id": "P12821",
        "disease_label": "Hypertension",
    },
    "Pneumonia": {
        "gene_symbol": "DEFB1",
        "protein_name": "DEFB1",
        "uniprot_id": "P60022",
        "disease_label": "Pneumonia",
    },
    "Hip Fracture": {
        "gene_symbol": "TNFSF11",
        "protein_name": "RANKL",
        "uniprot_id": "O14788",
        "disease_label": "Hip Fracture",
    },
    "Asthma": {
        "gene_symbol": "IL13",
        "protein_name": "IL13",
        "uniprot_id": "P35225",
        "disease_label": "Asthma",
    },
    "UTI": {
        "gene_symbol": "TLR4",
        "protein_name": "TLR4",
        "uniprot_id": "O00206",
        "disease_label": "UTI",
    },
    "Gallstones": {
        "gene_symbol": "ABCG8",
        "protein_name": "ABCG8",
        "uniprot_id": "Q9Y210",
        "disease_label": "Gallstones",
    },
    "Kidney Stones": {
        "gene_symbol": "SLC34A1",
        "protein_name": "SLC34A1",
        "uniprot_id": "Q06495",
        "disease_label": "Kidney Stones",
    },
    "Appendicitis": {
        "gene_symbol": "CRP",
        "protein_name": "CRP",
        "uniprot_id": "P02741",
        "disease_label": "Appendicitis",
    },
}
DEFAULT_PROTEIN_CONTEXT = {
    "gene_symbol": "TP53",
    "protein_name": "TP53",
    "uniprot_id": "P04637",
    "disease_label": "General Proteostasis",
}
MISFOLD_WEIGHTS = {
    "aggregation_propensity": 0.30,
    "surface_exposure_score": 0.25,
    "disorder_score": 0.15,
    "variant_delta_score": 0.15,
    "residue_graph_risk": 0.15,
}


class AggregationEvidenceProvider(ABC):
    @abstractmethod
    def get_aggregation_evidence(self, context: ProteinContext) -> dict:
        pass


class StructureFeatureProvider(ABC):
    @abstractmethod
    def get_structure_features(
        self,
        context: ProteinContext,
        dataset_overrides: dict | None = None,
    ) -> dict:
        pass


class VariantEffectProvider(ABC):
    @abstractmethod
    def get_variant_effect(self, context: ProteinContext) -> dict:
        pass


class FixtureAggregationEvidenceProvider(AggregationEvidenceProvider):
    def get_aggregation_evidence(self, context: ProteinContext) -> dict:
        baseline = _get_baseline_record(context)
        hotspot = _get_waltz_record(context)
        cpad_match = _find_cpad_match(context)

        evidence: list[dict] = []
        scores: list[float] = []

        if baseline:
            baseline_score = _clamp_score(baseline.get("baseline_aggregation_propensity"))
            if baseline_score is not None:
                scores.append(baseline_score)
            evidence.append({
                "source": "Protein Baseline Fixture",
                "type": "baseline_aggregation",
                "aggregation_propensity": baseline_score,
                "protein_name": baseline.get("protein_name"),
            })

        if hotspot:
            hotspot_score = _clamp_score(hotspot.get("aggregation_propensity"))
            if hotspot_score is not None:
                scores.append(hotspot_score)
            evidence.append({
                "source": "WALTZ-DB Fixture",
                "type": "hotspot_motif",
                "aggregation_propensity": hotspot_score,
                "motifs": hotspot.get("motifs", []),
                "hotspot_regions": hotspot.get("hotspot_regions", []),
            })

        if cpad_match:
            cpad_score = _clamp_score(float(cpad_match["aggregation_increase_pct"]) / 100.0)
            if cpad_score is not None:
                scores.append(cpad_score)
            evidence.append({
                "source": "CPAD Fixture",
                "type": "exact_variant_match",
                "protein_name": cpad_match.get("protein_name"),
                "variant_hgvs": cpad_match.get("variant_hgvs"),
                "aggregation_increase_pct": cpad_match.get("aggregation_increase_pct"),
                "evidence_level": cpad_match.get("evidence_level", "experimental"),
            })

        aggregation_propensity = round(max(scores), 4) if scores else None

        combined_hotspot_regions = _dedupe_list(
            (baseline or {}).get("hotspot_regions", []) +
            (hotspot or {}).get("hotspot_regions", [])
        )
        combined_surface_hotspots = _dedupe_list(
            (baseline or {}).get("surface_hotspots", []) +
            (hotspot or {}).get("surface_hotspots", [])
        )
        combined_windows = _dedupe_list(
            (baseline or {}).get("critical_residue_windows", []) +
            (hotspot or {}).get("critical_residue_windows", [])
        )

        return {
            "aggregation_propensity": aggregation_propensity,
            "aggregation_increase_pct": cpad_match.get("aggregation_increase_pct") if cpad_match else None,
            "exact_variant_match": cpad_match is not None,
            "hotspot_regions": combined_hotspot_regions,
            "surface_hotspots": combined_surface_hotspots,
            "critical_residue_windows": combined_windows,
            "evidence": evidence,
        }


class FixtureStructureFeatureProvider(StructureFeatureProvider):
    def get_structure_features(
        self,
        context: ProteinContext,
        dataset_overrides: dict | None = None,
    ) -> dict:
        baseline = _get_baseline_record(context) or _get_baseline_record(DEFAULT_PROTEIN_CONTEXT)
        dataset_overrides = dataset_overrides or {}

        baseline_surface = _clamp_score((baseline or {}).get("surface_exposure_score"))
        dataset_surface = _clamp_score(dataset_overrides.get("surface_exposure_score"))
        surface_exposure_score = dataset_surface if dataset_surface is not None else baseline_surface
        disorder_score = _clamp_score((baseline or {}).get("disorder_score"))
        residue_graph_risk = _clamp_score((baseline or {}).get("residue_graph_risk"))

        critical_region_ids = _dedupe_list(
            dataset_overrides.get("critical_region_ids", []) or
            (baseline or {}).get("hotspot_regions", [])
        )
        hotspot_residue_windows = _dedupe_list(
            (baseline or {}).get("critical_residue_windows", [])
        )
        surface_hotspots = _dedupe_list(
            (baseline or {}).get("surface_hotspots", [])
        )

        evidence = [{
            "source": "Protein Baseline Fixture",
            "type": "structure_profile",
            "surface_exposure_score": baseline_surface,
            "disorder_score": disorder_score,
            "residue_graph_risk": residue_graph_risk,
        }]
        if dataset_surface is not None:
            evidence.append({
                "source": "Dataset Override",
                "type": "surface_hydrophobic_ratio",
                "surface_exposure_score": dataset_surface,
            })
        if critical_region_ids:
            evidence.append({
                "source": "Dataset Override",
                "type": "critical_region_id",
                "critical_region_ids": critical_region_ids,
            })

        return {
            "surface_exposure_score": surface_exposure_score,
            "disorder_score": disorder_score,
            "residue_graph_risk": residue_graph_risk,
            "graph_summary": {
                "weak_edge_ratio": _clamp_score((baseline or {}).get("weak_edge_ratio")),
                "critical_region_ids": critical_region_ids,
                "hotspot_residue_windows": hotspot_residue_windows,
            },
            "hotspot_regions": _dedupe_list((baseline or {}).get("hotspot_regions", [])),
            "surface_hotspots": surface_hotspots,
            "critical_residue_windows": hotspot_residue_windows,
            "evidence": evidence,
        }


class FixtureVariantEffectProvider(VariantEffectProvider):
    def get_variant_effect(self, context: ProteinContext) -> dict:
        if not context.variant_hgvs:
            return {
                "variant_delta_score": None,
                "matched": False,
                "evidence": [],
            }

        match = _find_cpad_match(context)
        if not match:
            return {
                "variant_delta_score": None,
                "matched": False,
                "evidence": [{
                    "source": "CPAD Fixture",
                    "type": "no_curated_match",
                    "variant_hgvs": context.variant_hgvs,
                    "message": "No curated CPAD-derived evidence was available for this variant.",
                }],
            }

        return {
            "variant_delta_score": _clamp_score(match.get("variant_delta_score")),
            "matched": True,
            "evidence": [{
                "source": "CPAD Fixture",
                "type": "variant_delta",
                "variant_hgvs": match.get("variant_hgvs"),
                "variant_delta_score": _clamp_score(match.get("variant_delta_score")),
                "aggregation_increase_pct": match.get("aggregation_increase_pct"),
            }],
        }


def resolve_protein_context_for_run(
    run_id: str,
    explicit_context: ProteinContext | None = None,
) -> ProteinContext:
    explicit_data = explicit_context.model_dump(exclude_none=True) if explicit_context else {}
    dataset_context, _ = _extract_dataset_inputs(run_id)
    diagnosis_context = _resolve_diagnosis_context(run_id)

    merged = {
        **DEFAULT_PROTEIN_CONTEXT,
        **diagnosis_context,
        **dataset_context,
        **explicit_data,
    }
    hydrated = _hydrate_from_baseline(merged)
    return ProteinContext(**hydrated)


def resolve_protein_dataset_overrides(run_id: str) -> dict:
    _, dataset_overrides = _extract_dataset_inputs(run_id)
    return dataset_overrides


def misfold_tool(
    run_id: str,
    protein_context: ProteinContext,
    alphafold_result: dict,
    aggregation_provider: AggregationEvidenceProvider | None = None,
    structure_provider: StructureFeatureProvider | None = None,
    variant_provider: VariantEffectProvider | None = None,
) -> MisfoldSummary:
    aggregation_provider = aggregation_provider or FixtureAggregationEvidenceProvider()
    structure_provider = structure_provider or FixtureStructureFeatureProvider()
    variant_provider = variant_provider or FixtureVariantEffectProvider()

    dataset_overrides = resolve_protein_dataset_overrides(run_id)
    aggregation = aggregation_provider.get_aggregation_evidence(protein_context)
    structure = structure_provider.get_structure_features(protein_context, dataset_overrides)
    variant = variant_provider.get_variant_effect(protein_context)

    aggregation_propensity = aggregation.get("aggregation_propensity")
    surface_exposure_score = structure.get("surface_exposure_score")
    disorder_score = structure.get("disorder_score")
    variant_delta_score = variant.get("variant_delta_score")
    residue_graph_risk = structure.get("residue_graph_risk")

    component_values = {
        "aggregation_propensity": aggregation_propensity,
        "surface_exposure_score": surface_exposure_score,
        "disorder_score": disorder_score,
        "variant_delta_score": variant_delta_score,
        "residue_graph_risk": residue_graph_risk,
    }
    stuck_score = _weighted_score(component_values)
    energy_state = _map_energy_state(stuck_score)

    viewer_hotspot_regions = _dedupe_list(
        aggregation.get("hotspot_regions", []) +
        structure.get("hotspot_regions", []) +
        structure.get("graph_summary", {}).get("critical_region_ids", [])
    )
    viewer_stub = {
        "pdb_link": alphafold_result.get("pdb_link"),
        "hotspot_regions": viewer_hotspot_regions,
        "surface_hotspots": _dedupe_list(
            aggregation.get("surface_hotspots", []) +
            structure.get("surface_hotspots", [])
        ),
        "critical_residue_windows": _dedupe_list(
            aggregation.get("critical_residue_windows", []) +
            structure.get("critical_residue_windows", [])
        ),
        "graph_summary": structure.get("graph_summary", {}),
        "render_status": "placeholder",
    }

    evidence = (
        aggregation.get("evidence", []) +
        structure.get("evidence", []) +
        variant.get("evidence", [])
    )
    red_flags = _build_red_flags(
        protein_context=protein_context,
        aggregation=aggregation,
        surface_exposure_score=surface_exposure_score,
        energy_state=energy_state,
        stuck_score=stuck_score,
        hotspot_regions=viewer_hotspot_regions,
    )

    return MisfoldSummary(
        enabled=True,
        protein_name=protein_context.protein_name,
        uniprot_id=protein_context.uniprot_id,
        variant_hgvs=protein_context.variant_hgvs,
        aggregation_propensity=aggregation_propensity,
        stuck_score=stuck_score,
        energy_state=energy_state,
        surface_exposure_score=surface_exposure_score,
        disorder_score=disorder_score,
        variant_delta_score=variant_delta_score,
        residue_graph_risk=residue_graph_risk,
        evidence=evidence,
        red_flags=red_flags,
        viewer_stub=viewer_stub,
    )


def _resolve_diagnosis_context(run_id: str) -> dict:
    df = _load_run_dataframe(run_id)
    if df is None or "diagnosis" not in df.columns:
        return DEFAULT_PROTEIN_CONTEXT.copy()

    diagnosis = _series_mode_value(df["diagnosis"])
    if not diagnosis:
        return DEFAULT_PROTEIN_CONTEXT.copy()
    return DIAGNOSIS_PROTEIN_MAP.get(diagnosis, DEFAULT_PROTEIN_CONTEXT).copy()


def _extract_dataset_inputs(run_id: str) -> tuple[dict, dict]:
    df = _load_run_dataframe(run_id)
    if df is None:
        return {}, {}

    context: dict = {}
    for column in DATASET_CONTEXT_COLUMNS:
        if column not in df.columns:
            continue
        value = _series_mode_value(df[column])
        if value:
            context[column] = value

    overrides: dict = {}
    if "surface_hydrophobic_ratio" in df.columns:
        numeric = pd.to_numeric(df["surface_hydrophobic_ratio"], errors="coerce").dropna()
        if not numeric.empty:
            overrides["surface_exposure_score"] = round(float(numeric.mean()), 4)

    if "critical_region_id" in df.columns:
        critical_ids = [
            value for value in (_clean_value(v) for v in df["critical_region_id"].dropna().tolist())
            if value
        ]
        if critical_ids:
            overrides["critical_region_ids"] = _dedupe_list(critical_ids)

    return context, overrides


def _load_run_dataframe(run_id: str) -> pd.DataFrame | None:
    for filename in ("cleaned.csv", "ingested.csv"):
        path = os.path.join(settings.OUTPUT_DIR, run_id, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def _hydrate_from_baseline(context_data: dict) -> dict:
    baseline = _get_baseline_record(context_data)
    if not baseline:
        return context_data

    hydrated = context_data.copy()
    for key in ("gene_symbol", "protein_name", "uniprot_id"):
        hydrated[key] = baseline.get(key)
    if not hydrated.get("disease_label"):
        disease_labels = baseline.get("disease_labels", [])
        if disease_labels:
            hydrated["disease_label"] = disease_labels[0]
    return hydrated


def _build_red_flags(
    protein_context: ProteinContext,
    aggregation: dict,
    surface_exposure_score: float | None,
    energy_state: str | None,
    stuck_score: float | None,
    hotspot_regions: list[str],
) -> list[dict]:
    red_flags: list[dict] = []

    if aggregation.get("exact_variant_match") and (aggregation.get("aggregation_increase_pct") or 0) > 50:
        red_flags.append({
            "flag_id": "MF-001",
            "message": (
                f"Exact CPAD fixture match for {protein_context.variant_hgvs} indicates "
                f"{aggregation['aggregation_increase_pct']}% higher aggregation risk."
            ),
        })

    if energy_state in {"toxic_intermediate", "aggregation_prone"}:
        red_flags.append({
            "flag_id": "MF-002",
            "message": (
                f"Energy landscape suggests {energy_state} behavior "
                f"(stuck_score={stuck_score})."
            ),
        })

    if surface_exposure_score is not None and surface_exposure_score >= 0.65 and hotspot_regions:
        red_flags.append({
            "flag_id": "MF-003",
            "message": (
                f"Surface hydrophobic exposure is elevated ({surface_exposure_score}) "
                f"within hotspot regions {', '.join(hotspot_regions[:3])}."
            ),
        })

    return red_flags


def _weighted_score(component_values: dict[str, float | None]) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for key, value in component_values.items():
        if value is None:
            continue
        weight = MISFOLD_WEIGHTS[key]
        weighted_sum += value * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return round(weighted_sum / total_weight, 4)


def _map_energy_state(score: float | None) -> str | None:
    if score is None:
        return None
    if score < 0.30:
        return "native"
    if score < 0.55:
        return "strained"
    if score < 0.75:
        return "toxic_intermediate"
    return "aggregation_prone"


def _find_cpad_match(context: ProteinContext | dict) -> dict | None:
    context_dict = context if isinstance(context, dict) else context.model_dump(exclude_none=True)
    variant = _normalize(context_dict.get("variant_hgvs"))
    if not variant:
        return None

    gene_symbol = _normalize(context_dict.get("gene_symbol"))
    protein_name = _normalize(context_dict.get("protein_name"))
    uniprot_id = _normalize(context_dict.get("uniprot_id"))

    for row in _load_fixture("cpad_mutations.json"):
        if _normalize(row.get("variant_hgvs")) != variant:
            continue
        if gene_symbol and _normalize(row.get("gene_symbol")) == gene_symbol:
            return row
        if protein_name and _normalize(row.get("protein_name")) == protein_name:
            return row
        if uniprot_id and _normalize(row.get("uniprot_id")) == uniprot_id:
            return row
    return None


def _get_waltz_record(context: ProteinContext | dict) -> dict | None:
    baseline = _get_baseline_record(context)
    if not baseline:
        return None
    protein_name = baseline.get("protein_name")
    return _load_fixture("waltz_hotspots.json").get(protein_name)


def _get_baseline_record(context: ProteinContext | dict) -> dict | None:
    context_dict = context if isinstance(context, dict) else context.model_dump(exclude_none=True)
    baselines = _load_fixture("protein_baselines.json")

    search_terms = [
        _normalize(context_dict.get("uniprot_id")),
        _normalize(context_dict.get("gene_symbol")),
        _normalize(context_dict.get("protein_name")),
    ]
    for record in baselines.values():
        aliases = [_normalize(alias) for alias in record.get("aliases", [])]
        candidates = {
            _normalize(record.get("protein_name")),
            _normalize(record.get("gene_symbol")),
            _normalize(record.get("uniprot_id")),
            *aliases,
        }
        if any(term and term in candidates for term in search_terms):
            return record
    return None


@lru_cache(maxsize=8)
def _load_fixture(filename: str):
    with open(FIXTURE_DIR / filename, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _series_mode_value(series: pd.Series) -> str | None:
    cleaned_values = [_clean_value(value) for value in series.tolist()]
    cleaned_values = [value for value in cleaned_values if value]
    if not cleaned_values:
        return None

    mode = pd.Series(cleaned_values).mode()
    if not mode.empty:
        return str(mode.iloc[0])
    return str(cleaned_values[0])


def _clean_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize(value) -> str:
    return str(value or "").strip().lower()


def _clamp_score(value) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric < 0:
        return 0.0
    if numeric > 1:
        return 1.0
    return round(numeric, 4)


def _dedupe_list(values: list) -> list:
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
