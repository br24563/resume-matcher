import json
import os
import re

import anthropic
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

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


def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


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


def _map_api_error(exc: Exception) -> tuple[str, str, int]:
    if isinstance(exc, anthropic.RateLimitError):
        return (
            "Rate limit reached. Please wait a minute and try again.",
            "rate_limit",
            429,
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return (
            "Could not reach the Anthropic API. Check your internet connection and try again.",
            "network",
            502,
        )
    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "Invalid API key. Check ANTHROPIC_API_KEY in your .env file.",
            "auth",
            503,
        )
    if isinstance(exc, anthropic.BadRequestError):
        return (
            "The request was rejected by the API. Your input may be too long or malformed.",
            "bad_request",
            400,
        )
    if isinstance(exc, anthropic.APIError):
        return (f"Anthropic API error: {exc}", "api_error", 502)
    return (f"Unexpected error: {exc}", "unknown", 500)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    client = _get_client()
    if client is None:
        return _error_payload(
            "Server is missing ANTHROPIC_API_KEY. Add it to your .env file.",
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

    if not resume or not job_description:
        return _error_payload(
            "Both resume and job description are required.",
            "validation",
            400,
        )

    user_message = f"RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"

    def generate():
        full_text = []
        try:
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"

            raw = "".join(full_text)
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

        except anthropic.APIError as exc:
            message, error_type, _ = _map_api_error(exc)
            payload = {"type": "error", "error": message, "error_type": error_type}
            yield f"data: {json.dumps(payload)}\n\n"
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
