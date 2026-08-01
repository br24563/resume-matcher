import io
import json
import os
import re
import time
import base64
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from docx import Document
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq
from pypdf import PdfReader
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

# Rate limiter (no default limits; apply per-route)
limiter = Limiter(get_remote_address, app=app)

MAX_INPUT_CHARS = 4000
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
USAGE_LOG = "usage.log"

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


def _clean_extracted_text(text: str) -> str:
    """Normalize whitespace in extracted document text."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _extract_txt(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file.")


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _extract_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())


def _parse_model_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _error_payload(message: str, error_type: str, status: int = 500):
    return jsonify({"error": message, "error_type": error_type}), status


def _log_usage(endpoint: str, duration_ms: int):
    try:
        ts = datetime.now(timezone.utc).isoformat()
        # Do not log any user content — just timestamp, endpoint, duration
        with open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{endpoint}\t{duration_ms}\n")
    except Exception:
        pass


def _stream_analysis(client, system_prompt: str, user_message: str, model: str = None):
    def generate():
        start = time.monotonic()
        try:
            model_name = model or "llama-3.3-70b-versatile"
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or ""
            # log timing
            duration_ms = int((time.monotonic() - start) * 1000)
            _log_usage("/analyze", duration_ms)

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
def root_landing():
    # Serve landing page at root
    return send_from_directory("static", "landing.html")


@app.route("/app")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/extract-text", methods=["POST"])
@limiter.limit("10 per hour")
def extract_text():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Use .pdf, .docx, or .txt."}), 400

    data = file.read()
    if len(data) > MAX_FILE_SIZE:
        return jsonify({"error": "File exceeds 5MB limit."}), 400
    if not data:
        return jsonify({"error": "File is empty."}), 400

    try:
        if ext == ".txt":
            text = _extract_txt(data)
        elif ext == ".pdf":
            text = _extract_pdf(data)
        else:
            text = _extract_docx(data)
    except Exception as exc:
        return jsonify({"error": f"Could not extract text: {exc}"}), 422

    text = _clean_extracted_text(text)
    if not text:
        return jsonify({"error": "No text could be extracted from this file."}), 422

    return jsonify({"text": text})


@app.route("/scrape-job", methods=["POST"])
@limiter.limit("10 per hour")
def scrape_job():
    if not request.is_json:
        return _error_payload("Request must be JSON.", "validation", 400)
    data = request.get_json(silent=True)
    url = str(data.get("url", "")).strip()
    if not url:
        return _error_payload("URL is required.", "validation", 400)

    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "resume-matcher/1.0"})
        if resp.status_code != 200:
            return _error_payload("Couldn't fetch this page.", "fetch_error", 422)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove common boilerplate
        for sel in soup.select('nav, footer, header, script, style, aside'):
            sel.decompose()

        # Try to find title
        title = (soup.title.string if soup.title and soup.title.string else "").strip()
        if not title:
            h1 = soup.find(['h1'])
            if h1 and h1.get_text(strip=True):
                title = h1.get_text(strip=True)

        # Try to extract main content
        main = None
        main_tag = soup.find('main')
        if main_tag:
            main = main_tag.get_text(separator='\n').strip()
        if not main:
            # common job posting container heuristics
            candidates = soup.find_all(['div', 'section'])
            best = ""
            for c in candidates:
                text = c.get_text(separator='\n').strip()
                if len(text) > len(best):
                    best = text
            main = best

        # Heuristic: require substantial text
        if not main or len(main) < 200:
            return jsonify({"error": "Couldn't extract this page — paste the description manually"}), 422

        # Clean up whitespace
        main = re.sub(r"\n{3,}", "\n\n", main)
        main = re.sub(r"[ \t]+", " ", main).strip()

        return jsonify({"title": title, "description": main})
    except Exception:
        return jsonify({"error": "Couldn't extract this page — paste the description manually"}), 422


@app.route("/analyze", methods=["POST"])
@limiter.limit("10 per hour")
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
    model_field = str(data.get("model", "")).strip()

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

    # Map frontend selection to groq model strings
    model_map = {
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama-3.1-8b": "llama-3.1-8b-instant",
    }
    model = model_map.get(model_field, None)

    return _stream_analysis(client, system_prompt, user_message, model=model)


@app.route("/interview-prep", methods=["POST"])
@limiter.limit("10 per hour")
def interview_prep():
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
    resume = _sanitize_text(str(data.get("resume", "")))
    job_description = _sanitize_text(str(data.get("job_description", "")))
    model_field = str(data.get("model", "")).strip()

    if not resume or not job_description:
        return _error_payload("Both resume and job description are required.", "validation", 400)

    prompt = (
        "You are an interview coach. Given the resume and job description, generate 10 likely interview questions. "
        "For each question return a concise bullet-point answer framework (not a full scripted answer). Return ONLY valid JSON:"
        "{\"questions\":[{\"question\":\"...\",\"framework\":\"...\"}] }"
        f"\n\nRESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"
    )

    model_map = {
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama-3.1-8b": "llama-3.1-8b-instant",
    }
    model = model_map.get(model_field, None)

    try:
        model_name = model or "llama-3.3-70b-versatile"
        start = time.monotonic()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an interview coach. Provide JSON output as specified."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
        )
        raw = response.choices[0].message.content or ""
        duration_ms = int((time.monotonic() - start) * 1000)
        _log_usage("/interview-prep", duration_ms)
        parsed = _parse_model_json(raw)
        return jsonify(parsed)
    except json.JSONDecodeError:
        return _error_payload("The model returned malformed JSON. Please try again.", "parse_error", 502)
    except Exception as exc:
        return _error_payload(f"Unexpected error: {exc}", "unknown", 500)


@app.route("/stats", methods=["GET"])
def stats():
    today_count = 0
    all_count = 0
    total_ms = 0
    entries = 0
    today = datetime.now(timezone.utc).date()
    try:
        if os.path.exists(USAGE_LOG):
            with open(USAGE_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    ts_str, endpoint, ms_str = parts[0], parts[1], parts[2]
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        ms = int(ms_str)
                    except Exception:
                        continue
                    all_count += 1
                    total_ms += ms
                    entries += 1
                    if ts.date() == today:
                        today_count += 1
    except Exception:
        pass

    avg = int(total_ms / entries) if entries else 0
    return jsonify({"today": today_count, "all_time": all_count, "avg_response_ms": avg})


# Custom handler for rate limit errors to return consistent JSON
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Too many requests — try again in an hour"}), 429


if __name__ == "__main__":
    debug = os.environ.get("FLASK_ENV") != "production"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, host="0.0.0.0", port=port)
