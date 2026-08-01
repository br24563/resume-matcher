# Resume Matcher

Paste your resume and a job description, get back a match score, keyword gap analysis, and targeted bullet rewrites — powered by Claude.

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/your-username/resume-matcher
cd resume-matcher
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and replace `your_api_key_here` with your free API key from [console.groq.com → API Keys](https://console.groq.com/keys).

**3. Run**

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## How it works

- `app.py` — Flask server with a single `/analyze` endpoint. Reads your API key from `.env`, calls Claude, and returns structured JSON.
- `static/index.html` — Self-contained frontend. Posts to `/analyze` and renders the results.

## What you get

- **Match score** — 0–100% with an animated dial
- **Keywords found** — job description keywords already in your resume
- **Keywords to add** — important terms you're missing
- **Key gaps** — specific missing qualifications
- **Strengths** — what your resume does well for this role
- **Suggested rewrites** — before/after bullet improvements

## Deploying

For a public deployment, set `GROQ_API_KEY` as an environment variable on your host (Railway, Render, Fly.io, etc.) rather than using a `.env` file. The `.env` approach is for local development only.
