import os
import anthropic
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a professional resume analyst. Analyze how well a resume matches a job description.

Return ONLY valid JSON with this exact structure (no markdown, no preamble):
{
  "score": <integer 0-100>,
  "summary": "<2-sentence overall assessment>",
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
- score: integer 0-100 representing match percentage
- keywords_present: 5-10 important keywords from the job description that appear in the resume
- keywords_missing: 5-10 important keywords from the job description NOT in the resume
- gaps: 3-5 specific missing qualifications or experience areas
- strengths: 3-5 things the resume does well for this role
- rewrites: 3-4 specific bullet rewrites showing before/after to better target this job"""


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    resume = data.get("resume", "").strip()
    job_description = data.get("job_description", "").strip()

    if not resume or not job_description:
        return jsonify({"error": "Both resume and job description are required."}), 400

    # Truncate to avoid excessive token usage
    resume = resume[:4000]
    job_description = job_description[:4000]

    user_message = f"RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = message.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()

        import json
        result = json.loads(raw)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse model response. Please try again."}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"Anthropic API error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
