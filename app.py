import json
import os
import re

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

MAX_INPUT_CHARS = 4000

SYSTEM_PROMPT = """You are a professional resume analyst. Analyze how well a resume matches a job description.

Return ONLY valid JSON with this exact structure (no markdown, no preamble):
{
  "score": <integer 0-100>,
  "summary": "<2-sentence overall assessment>",
  "score_breakdown": {
    "keyword_match": {"score": <0-100>, "weight": 40, "detail": "<one sentence>"},
    "experience_alignment": {"score": <0-100>, "weight": 35, "detail": "<one sentence>"},
    "skills_coverage": {"score": <0-100>, "weight": 25, "detail": "<one sentence>"}
  },
  "keywords_present": ["<keyword>", ...],
  "keywords_missing": ["<keyword>", ...],
  "gaps": ["<gap description>", ...],
  "strengths": ["<strength>", ...],
  "rewrites": [
    {
      "section": "<e.g. Work Experience bullet>",
      "before": "<original text from resume, or describe what is missing>",
      "after": "<improved version tailored to this job>"
    }
  ]
}

Rules:
- score: integer 0-100 representing overall match percentage
- score_breakdown: component scores (0-100) with fixed weights (40/35/25); overall score should reflect these
- keywords_present: 5-10 important keywords from the job description that appear in the resume
- keywords_missing: 5-10 important keywords from the job description NOT in the resume
- gaps: 3-5 specific missing qualifications or experience areas
- strengths: 3-5 things the resume does well for this role
- rewrites: 3-4 specific bullet rewrites showing before/after to better target this job"""

ATS_SYSTEM_PROMPT = """You are an ATS (Applicant Tracking System) resume expert. Analyze how well a resume matches a job description AND how likely it is to pass automated ATS screening.

Return ONLY valid JSON with this exact structure (no markdown, no preamble):
{
  "score": <integer 0-100>,
  "ats_score": <integer 0-100>,
  "summary": "<2-sentence overall assessment covering both match quality and ATS readiness>",
  "score_breakdown": {
    "keyword_match": {"score": <0-100>, "weight": 40, "detail": "<one sentence>"},
    "experience_alignment": {"score": <0-100>, "weight": 35, "detail": "<one sentence>"},
    "skills_coverage": {"score": <0-100>, "weight": 25, "detail": "<one sentence>"}
  },
  "keywords_present": ["<keyword>", ...],
  "keywords_missing": ["<keyword>", ...],
  "gaps": ["<gap description>", ...],
  "strengths": ["<strength>", ...],
  "rewrites": [
    {
      "section": "<e.g. Work Experience bullet>",
      "before": "<original text>",
      "after": "<improved ATS-friendly version>"
    }
  ],
  "ats_analysis": {
    "keyword_density": {"score": <0-100>, "detail": "<assessment of keyword frequency and placement>"},
    "exact_phrase_match": {"score": <0-100>, "detail": "<how well exact job-description phrases appear verbatim>"},
    "section_headers": {"score": <0-100>, "detail": "<assessment>", "non_standard_headers": ["<creative header name found>", ...]},
    "parseability": {"score": <0-100>, "detail": "<tables, columns, graphics, special characters that break parsers>"},
    "format_recommendations": ["<recommendation>", ...]
  },
  "ats_fixes": [
    {"issue": "<specific ATS problem>", "fix": "<concrete fix>"}
  ]
}

Rules:
- score: integer 0-100 for overall job match (same criteria as standard analysis)
- ats_score: integer 0-100 for ATS compatibility alone (keyword density, headers, parseability, format)
- Focus on literal keyword matching — ATS systems match exact phrases, not synonyms
- Flag non-standard section headers (e.g. "Where I've Worked" instead of "Experience")
- Flag tables, multi-column layouts, text boxes, icons, and graphics that ATS parsers cannot read
- Recommend file format (prefer .docx or plain text over PDF with complex layout)
- keywords_present/missing: prioritize exact phrases from the job description
- gaps: include ATS-specific gaps (missing keywords, wrong headers)
- ats_fixes: 4-6 specific, actionable ATS improvements
- rewrites: show how to rephrase for both human readers and ATS keyword matching"""


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _sanitize_text(text: str) -> str:
    """Remove control characters and truncate."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
    return text[:MAX_INPUT_CHARS]


def _parse_model_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _error_payload(message: str, error_type: str, status: int = 500):
    return jsonify({"error": message, "error_type": error_type}), status


def _stream_analysis(client, system_prompt: str, user_message: str):
    def generate():
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or ""
            yield f"data: {json.dumps({'type': 'chunk', 'text': raw})}\n\n"

            try:
                result = _parse_model_json(raw)
            except json.JSONDecodeError:
                payload = {
                    "type": "error",
                    "error": "The model returned malformed JSON. Please try again.",
                    "error_type": "parse_error",
                }
                yield f"data: {json.dumps(payload)}\n\n"
                return

            yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

        except Exception as exc:
            payload = {
                "type": "error",
                "error": f"Unexpected error: {exc}",
                "error_type": "unknown",
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    client = _get_client()
    if client is None:
        return _error_payload(
            "Server is missing GROQ_API_KEY. Add it to your .env file.",
            "config",
            503,
        )

    if not request.is_json:
        return _error_payload("Request must be JSON.", "validation", 400)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error_payload("Invalid JSON body.", "validation", 400)

    resume = _sanitize_text(str(data.get("resume", "")))
    job_description = _sanitize_text(str(data.get("job_description", "")))
    mode = str(data.get("mode", "standard")).lower()

    if not resume or not job_description:
        return _error_payload(
            "Both resume and job description are required.",
            "validation",
            400,
        )

    if mode not in ("standard", "ats"):
        return _error_payload(
            'Mode must be "standard" or "ats".',
            "validation",
            400,
        )

    system_prompt = ATS_SYSTEM_PROMPT if mode == "ats" else SYSTEM_PROMPT
    user_message = f"RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"

    return _stream_analysis(client, system_prompt, user_message)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_ENV") != "production"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, host="0.0.0.0", port=port)
