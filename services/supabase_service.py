from supabase import create_client, Client
from utils.config import settings

OPTIONAL_RUN_FIELDS = {
    "protein_context_json",
    "protein_summary_json",
    "feature_summary",
}
OPTIONAL_INSIGHT_FIELDS = {
    "protein_summary_json",
    "doctor_report_text",
    "patient_report_text",
    "final_prescription_text",
    "report_status",
}


def get_supabase() -> Client:
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY
    )


def create_run_record(
    run_id: str,
    filename: str,
    filepath: str,
    problem_desc: str | None,
    target_column: str | None,
    protein_context_json: dict | None = None,
) -> None:
    supabase = get_supabase()
    row = {
        "run_id": run_id,
        "filename": filename,
        "filepath": filepath,
        "problem_desc": problem_desc,
        "target_column": target_column,
        "status": "queued",
    }
    if protein_context_json:
        row["protein_context_json"] = protein_context_json
    _safe_insert(supabase, "runs", row, OPTIONAL_RUN_FIELDS)


def update_run_status(run_id: str, **kwargs) -> None:
    supabase = get_supabase()
    _safe_update(supabase, "runs", kwargs, "run_id", run_id, OPTIONAL_RUN_FIELDS)


def insert_insight_row(row: dict) -> dict | None:
    supabase = get_supabase()
    result = _safe_insert(supabase, "insights", row, OPTIONAL_INSIGHT_FIELDS)
    if not result:
        return None
    return result.data[0] if result.data else None


def get_insight_by_run(run_id: str) -> dict | None:
    return fetch_single("insights", {"run_id": run_id})


def update_insight_by_id(insight_id: int, **kwargs) -> dict | None:
    supabase = get_supabase()
    result = _safe_update(
        supabase,
        "insights",
        kwargs,
        "id",
        insight_id,
        OPTIONAL_INSIGHT_FIELDS,
    )
    return result.data[0] if result and result.data else None


def update_insight_by_run(run_id: str, **kwargs) -> dict | None:
    supabase = get_supabase()
    result = _safe_update(
        supabase,
        "insights",
        kwargs,
        "run_id",
        run_id,
        OPTIONAL_INSIGHT_FIELDS,
    )
    return result.data[0] if result and result.data else None


def get_run(run_id: str) -> dict | None:
    return fetch_single("runs", {"run_id": run_id})


def fetch_rows(
    table: str,
    filters: dict | None = None,
    order_by: str | None = None,
    ascending: bool = False,
) -> list[dict]:
    supabase = get_supabase()
    try:
        query = supabase.table(table).select("*")
        for key, value in (filters or {}).items():
            query = query.eq(key, value)
        if order_by:
            query = query.order(order_by, desc=not ascending)
        result = query.execute()
        return result.data or []
    except Exception:
        return []


def fetch_single(table: str, filters: dict) -> dict | None:
    supabase = get_supabase()
    try:
        query = supabase.table(table).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        result = query.single().execute()
        return result.data
    except Exception:
        return None


def insert_row(
    table: str,
    row: dict,
    optional_fields: set[str] | None = None,
) -> dict | None:
    supabase = get_supabase()
    result = _safe_insert(supabase, table, row, optional_fields or set())
    if not result:
        return None
    return result.data[0] if result.data else None


def update_row(
    table: str,
    match_field: str,
    match_value,
    payload: dict,
    optional_fields: set[str] | None = None,
) -> dict | None:
    supabase = get_supabase()
    result = _safe_update(
        supabase,
        table,
        payload,
        match_field,
        match_value,
        optional_fields or set(),
    )
    return result.data[0] if result and result.data else None


def _safe_insert(
    supabase: Client,
    table: str,
    row: dict,
    optional_fields: set[str],
):
    try:
        return supabase.table(table).insert(row).execute()
    except Exception:
        fallback_row = {
            key: value for key, value in row.items()
            if key not in optional_fields
        }
        if fallback_row == row:
            raise
        return supabase.table(table).insert(fallback_row).execute()


def _safe_update(
    supabase: Client,
    table: str,
    payload: dict,
    match_field: str,
    match_value: str,
    optional_fields: set[str],
):
    try:
        return supabase.table(table).update(payload).eq(
            match_field, match_value
        ).execute()
    except Exception:
        fallback_payload = {
            key: value for key, value in payload.items()
            if key not in optional_fields
        }
        if fallback_payload == payload:
            raise
        return supabase.table(table).update(fallback_payload).eq(
            match_field, match_value
        ).execute()
