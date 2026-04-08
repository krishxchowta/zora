import polars as pl
import pandas as pd
import json
import os
from models.schemas import SchemaProfile
from utils.config import settings


def ingest_tool(
    filepath: str,
    run_id: str,
    target_column: str | None = None
) -> SchemaProfile:
    """
    Load file with Polars, profile schema, return SchemaProfile.
    Saves cleaned Pandas CSV to outputs/{run_id}/ingested.csv
    """
    # Load with Polars
    ext = filepath.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        lf = pl.scan_csv(filepath, infer_schema_length=5000)
    elif ext in ("xlsx", "xls"):
        lf = pl.from_pandas(
            pd.read_excel(filepath)
        ).lazy()
    elif ext == "json":
        lf = pl.scan_ndjson(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df = lf.collect()
    rows, cols = df.shape

    # Build column profiles
    columns = []
    numeric_cols, cat_cols, dt_cols = [], [], []

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        null_count = series.null_count()
        null_pct = round(null_count / rows * 100, 2) if rows else 0
        # sample up to 3 non-null values
        sample = (
            series.drop_nulls().head(3).to_list()
        )
        sample = [str(s) for s in sample]

        columns.append({
            "name": col,
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": null_pct,
            "sample_values": sample
        })

        if series.dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        ):
            numeric_cols.append(col)
        elif series.dtype == pl.Utf8:
            cat_cols.append(col)
        elif series.dtype in (pl.Date, pl.Datetime):
            dt_cols.append(col)

    # Heuristic target column detection
    target_candidate = target_column
    if not target_candidate:
        target_keywords = [
            "target", "label", "class", "outcome",
            "result", "churn", "diagnosis", "status"
        ]
        for col in df.columns:
            if any(kw in col.lower() for kw in target_keywords):
                target_candidate = col
                break

    # Null summary
    null_summary = {
        c["name"]: c["null_pct"]
        for c in columns
        if c["null_pct"] > 0
    }

    # Duplicate count
    dup_count = rows - df.unique().shape[0]

    # Memory
    memory_mb = round(df.estimated_size("mb"), 2)

    # Save Pandas CSV for downstream agents
    out_dir = os.path.join(settings.OUTPUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    df.to_pandas().to_csv(
        os.path.join(out_dir, "ingested.csv"), index=False
    )

    return SchemaProfile(
        run_id=run_id,
        filename=os.path.basename(filepath),
        rows=rows,
        cols=cols,
        columns=columns,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        datetime_columns=dt_cols,
        target_candidate=target_candidate,
        null_summary=null_summary,
        duplicate_count=dup_count,
        memory_mb=memory_mb
    )
