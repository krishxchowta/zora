from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from services.supabase_service import get_supabase
from utils.config import settings
from google import genai as google_genai
from google.genai import types as genai_types

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

@router.post("/query")
async def query_agent(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    supabase = get_supabase()
    
    # 1. Scope to a single demo patient
    # Fetch the latest run that has embeddings
    runs_res = supabase.table("runs").select("*").order("created_at", desc=True).limit(20).execute()
    runs = runs_res.data
    
    # Find the first run that has an embedding count > 0 (or fallback to the very first one)
    target_run_id = None
    if not runs:
        raise HTTPException(status_code=404, detail="No clinical runs found to search.")
        
    for run in runs:
        # Assuming the run has some indication it finished, or just pick the latest!
        if run.get("status") in ["queued", "running", "failed"]:
            continue
        target_run_id = run["run_id"]
        break
        
    if not target_run_id:
        target_run_id = runs[0]["run_id"] # Fallback

    # 2. Embed the user's query
    client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    try:
        emb_res = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[request.query],
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        query_embedding = emb_res.embeddings[0].values
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {e}")

    # 3. Fetch documents for this run
    docs_res = supabase.table("documents").select("*").eq("run_id", target_run_id).execute()
    documents = docs_res.data
    
    if not documents:
        # If no documents, we can still ask Gemini to answer based on general knowledge or state ignorance
        top_chunks_text = "No clinical documents found for this patient."
    else:
        # 4. Perform vector similarity search in Python
        scored_docs = []
        for doc in documents:
            doc_emb = doc.get("embedding")
            if not doc_emb:
                continue
            sim = cosine_similarity(query_embedding, doc_emb)
            scored_docs.append((sim, doc))
            
        # Sort by similarity desc
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Take Top-5 chunks
        top_k = 5
        top_chunksText = []
        for sim, doc in scored_docs[:top_k]:
            top_chunksText.append(f"Document Chunk: {doc.get('chunk_text', '')}")
            
        top_chunks_text = "\n\n".join(top_chunksText)

    # 5. Query Gemini 2.5 Flash
    system_prompt = (
        "You are a clinical AI assistant for a physician dashboard explicitly focused on ONE patient's data.\n"
        "Use the provided clinical dataset contexts to answer the user's question accurately.\n"
        "If the answer is not in the context, clearly state that you do not know based on the provided documents.\n\n"
        f"--- CLINICAL CONTEXT START ---\n{top_chunks_text}\n--- CLINICAL CONTEXT END ---"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt + f"\n\nQuestion: {request.query}"
        )
        answer_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    return {
        "answer": answer_text,
        "run_id_scoped": target_run_id
    }
