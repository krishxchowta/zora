import re


PROTEIN_SIDECAR_COLUMNS = {
    "gene_symbol",
    "protein_name",
    "uniprot_id",
    "variant_hgvs",
    "surface_hydrophobic_ratio",
    "critical_region_id",
    "disease_label",
}


def normalize_column_name(name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "column"


def normalize_column_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    normalized: list[str] = []

    for column in columns:
        base = normalize_column_name(column)
        count = seen.get(base, 0)
        seen[base] = count + 1
        normalized.append(base if count == 0 else f"{base}_{count + 1}")

    return normalized


def normalize_target_column(
    target_column: str | None,
    available_columns: list[str],
) -> str | None:
    if target_column and target_column in available_columns:
        return target_column

    if target_column:
        normalized = normalize_column_name(target_column)
        if normalized in available_columns:
            return normalized
        return normalized

    if "readmission_30day" in available_columns:
        return "readmission_30day"

    return None
