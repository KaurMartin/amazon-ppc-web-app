# Amazon PPC Negative Keyword Analyzer Web App

This is a ready-to-run Streamlit web app for analyzing Amazon PPC search term reports and surfacing reusable negative keyword candidates.

## What it does

- accepts one or more Amazon Search Term CSV files
- accepts an optional Negative Keyword CSV file
- detects Amazon column names using aliases instead of fixed column positions
- analyzes both single words and adjacent word pairs
- scores low-relevance candidates using campaign-relative logic
- excludes terms already present in the uploaded negative keyword list
- lets you download each output CSV or a full ZIP of all outputs

## Files

- `app.py` — Streamlit webpage
- `analyzer_core.py` — reusable analysis engine
- `requirements.txt` — Python dependencies
- `run_app.sh` — shortcut script to launch locally

## Local setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

Or:

```bash
./run_app.sh
```

## What to upload in the app

### Required
- one or more Amazon Sponsored Products Search Term report CSVs

### Optional
- one Amazon Negative Keyword report CSV

## Outputs

- `00_summary.json`
- `01_cleaned_search_terms.csv`
- `02_negative_keywords_cleaned.csv`
- `03_single_word_analysis.csv`
- `04_word_pair_analysis.csv`
- `05_recommended_negative_candidates.csv`
- `06_high_click_zero_sale_search_terms.csv`

## Deploy as a webpage

### Streamlit Community Cloud
1. Upload this folder to a GitHub repository
2. Create a new Streamlit app
3. Set the main file to `app.py`
4. Deploy

### Render / Railway / VPS
Use:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Recommended use

- run weekly after downloading fresh Amazon reports
- review recommendations before adding negatives in Amazon Ads
- keep thresholds conservative at first
- adjust thresholds based on account size and traffic volume

## Important note

This app is intentionally performance-based and campaign-agnostic. It helps identify likely negative candidates, but it does not understand product semantics on its own. Final manual review is still recommended before applying negatives.
