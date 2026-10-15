# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get:

- an **ATS score out of 100** shown at the top
- a **score breakdown** (keywords, content impact, formatting, structure, readability)
- **prioritised improvements** (High / Medium / Low) with how to fix each one
- **found vs. missing keywords** (paste a job description for a tailored match)
- **bullet-point rewrites** and a downloadable report

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

## How the score works

Gemini rates five categories from 0–100. The overall score is calculated **in code** with fixed
weights, so it stays consistent:

| Category | Weight |
|---|---|
| Keywords & relevance | 30% |
| Content & impact | 25% |
| Formatting & parseability | 20% |
| Structure & sections | 15% |
| Readability & language | 10% |

> This is an AI estimate, not the output of a real employer ATS (Workday, Greenhouse, etc.
> each behave differently). Use it as guidance for improving your resume.

## Run locally

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Provide your key (pick one):

```bash
# Option A – environment variable
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# Option B – Streamlit secrets file (recommended)
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml
```

   Or just paste the key into the app's sidebar.

4. Start the app:

```bash
streamlit run app.py
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | – |
| `GEMINI_MODEL` | secrets / env var / sidebar | `gemini-flash-latest` |

`gemini-flash-latest` always points to Google's newest Flash model. To pin a version, set
`GEMINI_MODEL` (e.g. `gemini-2.5-flash`).

## Deploy to Streamlit Community Cloud

1. Push this project to GitHub (see below). **Never commit `.streamlit/secrets.toml`.**
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**. You get a public `*.streamlit.app` URL in a couple of minutes.

## Project structure

```
app.py            # Streamlit UI + Gemini analysis
requirements.txt  # Python dependencies
README.md
```

## Notes & limitations

- Scanned / image-only PDFs have no extractable text – export a text-based PDF instead.
- Resume text is sent to the Gemini API; don't upload documents you aren't comfortable sharing.
- Files are limited to 5 MB and about 30,000 characters of text.
