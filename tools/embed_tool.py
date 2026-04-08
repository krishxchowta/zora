from google import genai as google_genai
from google.genai import types as genai_types
from langchain_core.documents import Document
from supabase import create_client
from models.schemas import SchemaProfile
from utils.config import settings
import json


def build_chunks(profile: SchemaProfile) -> list[Document]:
    """
    Convert SchemaProfile into embeddable text chunks.
    Each chunk is a Document with run_id in metadata.
    """
    chunks = []

    # Chunk 0 — dataset overview
    overview = (
        f"Dataset: {profile.filename}. "
        f"Rows: {profile.rows}. Columns: {profile.cols}. "
        f"Memory: {profile.memory_mb}MB. "
        f"Duplicate rows: {profile.duplicate_count}. "
        f"Target column candidate: {profile.target_candidate or 'not detected'}. "
        f"Numeric columns: {', '.join(profile.numeric_columns)}. "
        f"Categorical columns: {', '.join(profile.categorical_columns)}. "
        f"Datetime columns: {', '.join(profile.datetime_columns) or 'none'}."
    )
    chunks.append(Document(
        page_content=overview,
        metadata={
            "run_id": profile.run_id,
            "chunk_type": "overview",
            "chunk_index": 0
        }
    ))

    # Chunk 1 — null quality summary
    if profile.null_summary:
        null_text = "Missing value summary: " + ". ".join(
            f"{col} is {pct}% null"
            for col, pct in profile.null_summary.items()
        ) + "."
    else:
        null_text = "No missing values detected in any column."
    chunks.append(Document(
        page_content=null_text,
        metadata={
            "run_id": profile.run_id,
            "chunk_type": "null_summary",
            "chunk_index": 1
        }
    ))

    # One chunk per column
    for i, col_info in enumerate(profile.columns):
        col_text = (
            f"Column '{col_info['name']}': "
            f"type {col_info['dtype']}, "
            f"{col_info['null_pct']}% null. "
            f"Sample values: {', '.join(col_info['sample_values'])}."
        )
        chunks.append(Document(
            page_content=col_text,
            metadata={
                "run_id": profile.run_id,
                "chunk_type": "column_profile",
                "chunk_index": i + 2,
                "column_name": col_info["name"]
            }
        ))

    return chunks


def embed_tool(profile: SchemaProfile) -> int:
    """
    Embed all schema chunks into Supabase pgvector using
    google-genai SDK directly (avoids v1beta API issues in
    langchain-google-genai).
    Returns count of vectors stored.
    """
    chunks = build_chunks(profile)
    texts = [c.page_content for c in chunks]

    # Generate embeddings — gemini-embedding-001 with 768-dim output
    # to match the vector(768) Supabase schema
    client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )

    # Build rows for Supabase insert
    rows = []
    for chunk, emb in zip(chunks, response.embeddings):
        rows.append({
            "run_id": profile.run_id,
            "chunk_index": chunk.metadata["chunk_index"],
            "chunk_text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": emb.values
        })

    # Insert into Supabase documents table
    supabase = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY
    )
    supabase.table("documents").insert(rows).execute()

    return len(rows)
