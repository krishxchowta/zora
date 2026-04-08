"""
S6 — Zora Narrator Agent

Dual-voice narration:
  - clinical_mode : dense technical paragraph (model, AUC, SHAP, protein, denial %, waste $, safety rule IDs)
  - patient_mode  : plain English ("Your test results show…", treatment + cost)

[G2] Critic Gate 2 — quality check:
  Scores both voices on clarity (1-5), completeness (1-5), tone (1-5).
  Composite = average of per-voice sums. Pass if composite >= 7/10.
  One retry with feedback if G2 fails.

Optional delivery:
  - Twilio SMS : sends narration_patient if TWILIO_ACCOUNT_SID is configured
  - Google TTS : synthesizes narration_patient as MP3 if CLOUD_TTS_API_KEY is configured

Writes narration_clinical, narration_patient, g2_score, g2_passed to insights table.
Emits pipeline_complete (status=full_complete) to close the SSE stream.
"""

import os
import time
import json
import base64
import requests
from crewai import Agent, Task, Crew, Process, LLM

from services.supabase_service import update_insight_by_id, update_run_status
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import SchemaProfile
from datetime import datetime, timezone

G2_PASS_THRESHOLD = 7.0
MAX_NARRATOR_RETRIES = 2


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _narrator_kickoff(prompt: str, expected_output: str) -> str:
    """Run narrator with Gemini → Groq fallback."""
    candidates = [
        LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=0.3),
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0.3),
    ]
    last_exc = None
    for llm in candidates:
        try:
            agent = Agent(
                role="Healthcare Narrator",
                goal=(
                    "Write clear, accurate narrations of clinical analytics results "
                    "for two distinct audiences: clinicians and patients."
                ),
                backstory=(
                    "Expert medical communicator with deep knowledge of clinical ML, "
                    "health economics, and patient education."
                ),
                llm=llm, verbose=False, allow_delegation=False, max_iter=1
            )
            task = Task(description=prompt, expected_output=expected_output, agent=agent)
            return str(Crew(agents=[agent], tasks=[task],
                            process=Process.sequential, verbose=False).kickoff())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Narrator LLM failed: {last_exc}")


def _g2_critic_kickoff(prompt: str) -> dict:
    """Run G2 Critic with llama-3.1-8b-instant → llama-3.3-70b-versatile fallback."""
    candidates = [
        LLM(model="groq/llama-3.1-8b-instant", api_key=settings.GROQ_API_KEY, temperature=0.0),
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0.0),
    ]
    last_exc = None
    for llm in candidates:
        try:
            agent = Agent(
                role="Narration Quality Critic",
                goal=(
                    "Evaluate two healthcare narrations (clinical and patient mode) "
                    "on clarity, completeness, and tone. "
                    "Return ONLY a JSON object."
                ),
                backstory=(
                    "Expert medical editor who ensures clinical accuracy and patient accessibility "
                    "in healthcare communications."
                ),
                llm=llm, verbose=False, allow_delegation=False, max_iter=1
            )
            task = Task(
                description=prompt,
                expected_output=(
                    'Valid JSON only. Example: '
                    '{"clinical_clarity": 4, "clinical_completeness": 4, "clinical_tone": 4, '
                    '"patient_clarity": 4, "patient_completeness": 4, "patient_tone": 4, '
                    '"feedback": "Both narrations are accurate and well-structured."}'
                ),
                agent=agent
            )
            raw = str(Crew(agents=[agent], tasks=[task],
                           process=Process.sequential, verbose=False).kickoff()).strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"G2 Critic LLM failed: {last_exc}")


def _compute_g2_score(g2_result: dict) -> float:
    """Average the two per-voice sums (max 15 each) normalised to 10."""
    clinical_sum = (
        g2_result.get("clinical_clarity", 3)
        + g2_result.get("clinical_completeness", 3)
        + g2_result.get("clinical_tone", 3)
    )
    patient_sum = (
        g2_result.get("patient_clarity", 3)
        + g2_result.get("patient_completeness", 3)
        + g2_result.get("patient_tone", 3)
    )
    avg_sum = (clinical_sum + patient_sum) / 2          # avg of two voices [3-15]
    return round((avg_sum / 15) * 10, 2)                # normalize to [2-10]


# ── Optional delivery ─────────────────────────────────────────────────────────

def _send_twilio_sms(body: str, to_number: str) -> bool:
    """Send SMS via Twilio REST. Returns True on success."""
    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=body[:1600],   # SMS limit
            from_=settings.TWILIO_SMS_FROM,
            to=to_number
        )
        return msg.sid is not None
    except Exception:
        return False


def _synthesize_tts(text: str, run_id: str) -> str | None:
    """
    Call Google Cloud TTS REST API to synthesize text as MP3.
    Saves to outputs/{run_id}/narration_patient.mp3.
    Returns file path on success, None on failure.
    """
    try:
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={settings.CLOUD_TTS_API_KEY}"
        payload = {
            "input": {"text": text[:5000]},
            "voice": {
                "languageCode": "en-IN",
                "name": settings.CLOUD_TTS_VOICE_EN,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": float(settings.CLOUD_TTS_SPEAKING_RATE),
            }
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        audio_b64 = resp.json().get("audioContent", "")
        if not audio_b64:
            return None

        out_dir = os.path.join(settings.OUTPUT_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "narration_patient.mp3")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        return out_path
    except Exception:
        return None


# ── Main agent ────────────────────────────────────────────────────────────────

async def run_narrator_agent(
    run_id: str,
    profile: SchemaProfile,
    synthesis_result: dict,
    phone_number: str | None = None,
) -> dict:
    """
    Generate dual-voice narrations, run G2 critic, optionally deliver via SMS/TTS.
    Writes to insights table. Emits pipeline_complete to close SSE stream.

    Args:
        run_id:           pipeline run identifier
        profile:          SchemaProfile from S1
        synthesis_result: dict from run_synthesis_agent (insight_id, synthesis_text,
                          finance, safety, rag_citations)
        phone_number:     optional E.164 number for Twilio SMS delivery

    Returns:
        {"narration_clinical": str, "narration_patient": str,
         "g2_score": float, "g2_passed": bool}
    """
    t0 = time.monotonic()

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_narrator",
        "status": "running",
        "output_summary": "Generating clinical and patient narrations...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    insight_id      = synthesis_result.get("insight_id")
    synthesis_text  = synthesis_result.get("synthesis_text", "")
    finance         = synthesis_result.get("finance", {})
    safety          = synthesis_result.get("safety", {})
    rag_citations   = synthesis_result.get("rag_citations", [])
    misfold         = synthesis_result.get("misfold", {})

    safety_flags = safety.get("safety_flags", [])
    safety_flag_str = (
        "; ".join(f"[{f['rule_id']}] {f['message']}" for f in safety_flags)
        or "No safety flags triggered."
    )
    rag_ref_str = (
        "\n".join(f"- {c['chunk_text'][:120]}..." for c in rag_citations[:3])
        or "No RAG citations available."
    )
    misfold_enabled = bool(misfold and misfold.get("enabled"))
    misfold_summary = ""
    patient_misfold_summary = ""
    if misfold_enabled:
        variant_delta = misfold.get("variant_delta_score")
        variant_delta_text = (
            f"{variant_delta:.2f}" if isinstance(variant_delta, (int, float))
            else "not available from curated evidence"
        )
        misfold_summary = f"""

MISFOLD RISK:
  Stuck-score: {misfold.get('stuck_score')} ({misfold.get('energy_state')})
  Aggregation propensity: {misfold.get('aggregation_propensity')}
  Surface exposure: {misfold.get('surface_exposure_score')}
  Variant delta score: {variant_delta_text}
  Hotspot regions: {', '.join(misfold.get('viewer_stub', {}).get('hotspot_regions', [])[:3]) or 'none'}
""".rstrip()
        patient_misfold_summary = f"""

PROTEIN CLUMPING RISK:
  Risk state: {misfold.get('energy_state')}
  Stuck-score: {misfold.get('stuck_score')}
""".rstrip()

    # ── Build narrator prompts ────────────────────────────────────────────────
    clinical_prompt = f"""
Write a CLINICAL narration (dense technical, 5-7 sentences) for a physician audience.

SYNTHESIS: {synthesis_text}

FINANCIAL RISK:
  Denial probability: {finance.get('denial_probability', 0)*100:.1f}%
  Waste estimate: ${finance.get('waste_estimate_usd', 0):,.0f}
  Predicted readmission rate: {finance.get('predicted_readmission_rate', 0)*100:.0f}%

SAFETY FLAGS: {safety_flag_str}
DOCTOR REVIEW REQUIRED: {safety.get('doctor_review', False)}
{misfold_summary}

RAG EVIDENCE:
{rag_ref_str}

Requirements:
- Include model name, AUC, accuracy, F1 metrics
- Cite top SHAP features by name
- Reference protein stability score and confidence level
- Include misfold/aggregation risk if present
- Include denial probability and waste estimate with dollar figures
- Cite safety rule IDs (SR-001, SR-002, etc.) where triggered
- End with specific clinical action recommendation
""".strip()

    patient_prompt = f"""
Write a PATIENT narration (plain English, 4-6 sentences) for a patient/family audience.

SYNTHESIS: {synthesis_text}

FINANCIAL RISK:
  Denial probability: {finance.get('denial_probability', 0)*100:.1f}%
  Estimated cost impact: ${finance.get('waste_estimate_usd', 0):,.0f}
  Readmission risk level: {finance.get('predicted_readmission_rate', 0)*100:.0f}%

DOCTOR REVIEW: {safety.get('doctor_review', False)}
{patient_misfold_summary}

Requirements:
- Start with "Your health analysis shows..."
- Use plain language — no acronyms, no jargon
- Explain what the risk percentage means in everyday terms
- Mention if a doctor review is required and why it matters
- Mention protein clumping risk only if present, in non-technical language
- Give one actionable step the patient can take (follow-up appointment, lifestyle change, etc.)
- Keep a reassuring, empathetic tone
""".strip()

    # ── Generate narrations (with one G2-feedback retry) ─────────────────────
    narration_clinical = ""
    narration_patient  = ""
    g2_result: dict    = {}
    g2_score           = 0.0
    g2_passed          = False
    g2_feedback        = ""

    for attempt in range(1, MAX_NARRATOR_RETRIES + 1):

        # Append feedback to prompt on retry
        retry_note = f"\n\nPREVIOUS FEEDBACK: {g2_feedback}\nPlease address the above issues." if g2_feedback else ""

        narration_clinical = _narrator_kickoff(
            clinical_prompt + retry_note,
            "A dense clinical paragraph 5-7 sentences covering ML metrics, SHAP features, "
            "protein stability, financial risk, safety rules, and clinical recommendation."
        )
        narration_patient = _narrator_kickoff(
            patient_prompt + retry_note,
            "A plain English paragraph 4-6 sentences starting with 'Your health analysis shows...' "
            "covering risk level, doctor review need, and one actionable step."
        )

        # ── G2 Critic Gate 2 ─────────────────────────────────────────────────
        await sse_manager.publish(run_id, {
            "type": "agent_update",
            "agent": "zora_critic_g2",
            "status": "running",
            "output_summary": f"G2 Critic evaluating narration quality (attempt {attempt}/{MAX_NARRATOR_RETRIES})...",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        g2_prompt = f"""
Evaluate the quality of two healthcare narrations. Score each dimension 1-5.

CLINICAL NARRATION:
{narration_clinical}

PATIENT NARRATION:
{narration_patient}

Score each voice on:
- clarity      (1=confusing, 5=crystal clear)
- completeness (1=missing key info, 5=covers all relevant points)
- tone         (1=inappropriate, 5=perfectly pitched for audience)

Return ONLY valid JSON:
{{
  "clinical_clarity": <1-5>,
  "clinical_completeness": <1-5>,
  "clinical_tone": <1-5>,
  "patient_clarity": <1-5>,
  "patient_completeness": <1-5>,
  "patient_tone": <1-5>,
  "feedback": "<one sentence summary of main issues, if any>"
}}
""".strip()

        g2_result  = _g2_critic_kickoff(g2_prompt)
        g2_score   = _compute_g2_score(g2_result)
        g2_passed  = g2_score >= G2_PASS_THRESHOLD
        g2_feedback = g2_result.get("feedback", "")

        await sse_manager.publish(run_id, {
            "type": "agent_update",
            "agent": "zora_critic_g2",
            "status": "completed" if g2_passed else "warning",
            "output_summary": (
                f"G2: {'PASS' if g2_passed else 'WARN'} score={g2_score}/10. "
                f"{g2_feedback}"
            ),
            "data": {
                "g2_score":   g2_score,
                "g2_passed":  g2_passed,
                "attempt":    attempt,
                "scores":     {k: v for k, v in g2_result.items() if k != "feedback"},
                "feedback":   g2_feedback,
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if g2_passed or attempt == MAX_NARRATOR_RETRIES:
            break

    # ── Optional Twilio SMS ───────────────────────────────────────────────────
    sms_sent = False
    if phone_number and getattr(settings, "TWILIO_ACCOUNT_SID", None):
        sms_sent = _send_twilio_sms(narration_patient, phone_number)

    # ── Optional Google TTS ───────────────────────────────────────────────────
    tts_path = None
    if getattr(settings, "CLOUD_TTS_API_KEY", None):
        tts_path = _synthesize_tts(narration_patient, run_id)

    # ── Write narrator results to insights table ──────────────────────────────
    if insight_id:
        update_insight_by_id(insight_id, **{
            "narration_clinical": narration_clinical,
            "narration_patient":  narration_patient,
            "doctor_report_text": narration_clinical,
            "patient_report_text": narration_patient,
            "g2_score":           g2_score,
            "g2_passed":          g2_passed,
            "report_status":      "draft_available",
        })

    latency_ms = int((time.monotonic() - t0) * 1000)

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_narrator",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"Narrations written. G2 score: {g2_score}/10 ({'PASS' if g2_passed else 'WARN'}). "
            f"{'SMS sent. ' if sms_sent else ''}"
            f"{'TTS saved. ' if tts_path else ''}"
            f"Insight #{insight_id} updated."
        ),
        "data": {
            "insight_id":         insight_id,
            "g2_score":           g2_score,
            "g2_passed":          g2_passed,
            "sms_sent":           sms_sent,
            "tts_path":           tts_path,
            "clinical_length":    len(narration_clinical),
            "patient_length":     len(narration_patient),
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Update run status ─────────────────────────────────────────────────────
    update_run_status(
        run_id,
        status="full_complete",
        completed_at=datetime.now(timezone.utc).isoformat()
    )

    # ── pipeline_complete — closes the SSE stream ─────────────────────────────
    await sse_manager.publish(run_id, {
        "type": "pipeline_complete",
        "agent": "zora_narrator",
        "run_id": run_id,
        "status": "full_complete",
        "output_summary": (
            f"Full pipeline complete (S1→S5+Narrator). "
            f"Insight #{insight_id}. "
            f"G2 score: {g2_score}/10."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return {
        "narration_clinical": narration_clinical,
        "narration_patient":  narration_patient,
        "g2_score":           g2_score,
        "g2_passed":          g2_passed,
    }
