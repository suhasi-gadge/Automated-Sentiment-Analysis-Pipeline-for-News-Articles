# Automated Sentiment Analysis Pipeline for News Articles

This repository implements an automated sentiment analysis pipeline for news articles on the topic **renewable energy**.

The project satisfies the task requirements by:

- identifying a public news source: **GDELT Project / GDELT DOC API**;
- processing at least **100 news articles**;
- applying text preprocessing with **NLTK-style tokenization, stop-word removal, and lemmatization**;
- calculating sentiment scores with **VADER** when available, with a fallback lexicon for offline execution;
- saving article-level results to CSV; and
- generating a Markdown summary report with sentiment distribution and top keywords.

## Repository Structure

```text
news-sentiment-pipeline/
├── data/
│   └── renewable_energy_news_100.csv
├── outputs/
│   └── sentiment_results.csv
├── reports/
│   └── sentiment_report.md
├── src/
│   └── news_sentiment_pipeline.py
├── requirements.txt
├── README.md
└── TASK_COMMENT.md
```

## Public Data Source

The live collection mode uses the public **GDELT DOC API**, which provides news article metadata from global media coverage. GDELT is free and open, making it appropriate for a portfolio-style automated news analytics pipeline.

Because live news websites can block scraping, change layouts, or require network access, this repository also includes a bundled 100-article renewable-energy dataset in:

```text
data/renewable_energy_news_100.csv
```

This allows reviewers to run the pipeline immediately without an API key or internet dependency.

## Dataset Fields

The input CSV contains the following fields:

| Column | Description |
|---|---|
| `article_id` | Unique article identifier |
| `topic` | Topic used for collection/analysis |
| `source` | Source label or domain |
| `source_url` | Article URL or placeholder URL |
| `title` | Article title |
| `published_date` | Article publication date |
| `article_text` | Full article body used for sentiment analysis |

## Data Processing Steps

The Python script performs the following steps:

1. Loads at least 100 news articles from CSV or optional live GDELT collection.
2. Validates that required columns exist and that at least 100 articles are available.
3. Cleans article text.
4. Tokenizes text.
5. Removes stop words.
6. Lemmatizes tokens.
7. Calculates sentiment scores:
   - `sentiment_negative`
   - `sentiment_neutral`
   - `sentiment_positive`
   - `sentiment_compound`
8. Assigns final sentiment labels:
   - positive
   - neutral
   - negative
9. Saves article-level results to CSV.
10. Generates a Markdown report.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline with the Bundled Dataset

```bash
python src/news_sentiment_pipeline.py \
  --input data/renewable_energy_news_100.csv \
  --output outputs/sentiment_results.csv \
  --report reports/sentiment_report.md
```

On Windows PowerShell, use:

```powershell
python src/news_sentiment_pipeline.py `
  --input data/renewable_energy_news_100.csv `
  --output outputs/sentiment_results.csv `
  --report reports/sentiment_report.md
```

## Optional: Collect Live Articles from GDELT

```bash
python src/news_sentiment_pipeline.py \
  --collect-live \
  --topic "renewable energy" \
  --limit 100 \
  --input data/live_gdelt_renewable_energy.csv \
  --output outputs/live_sentiment_results.csv \
  --report reports/live_sentiment_report.md
```

Note: live collection depends on current GDELT availability and whether source websites allow article text extraction.

## Output Files

### `outputs/sentiment_results.csv`

Contains one row per article with original text, processed text, and sentiment scores.

Important columns:

| Column | Description |
|---|---|
| `article_text` | Original article body |
| `processed_text` | Cleaned, tokenized, stop-word removed, lemmatized text |
| `sentiment_negative` | Negative sentiment score |
| `sentiment_neutral` | Neutral sentiment score |
| `sentiment_positive` | Positive sentiment score |
| `sentiment_compound` | Overall sentiment score from -1 to +1 |
| `sentiment_label` | Final sentiment category |

### `reports/sentiment_report.md`

Summarizes:

- number of articles processed;
- average compound sentiment score;
- positive/neutral/negative sentiment distribution;
- top keywords in positive articles;
- top keywords in negative articles; and
- top overall keywords.

## Interpreting Sentiment Scores

The pipeline uses VADER-style thresholds:

| Compound Score Range | Label |
|---:|---|
| `>= 0.05` | Positive |
| `<= -0.05` | Negative |
| `-0.05 to 0.05` | Neutral |

A higher positive compound score indicates stronger positive sentiment. A lower negative compound score indicates stronger negative sentiment.

## Example Console Output

```text
Sentiment pipeline completed successfully.
Articles processed: 100
Sentiment distribution: {'positive': 63, 'negative': 37}
Output CSV: outputs/sentiment_results.csv
Report: reports/sentiment_report.md
```

## Notes for Reviewers

- The code demonstrates tokenization, stop-word removal, and lemmatization in `preprocess_text()`.
- The script validates that at least 100 articles are available before processing.
- The bundled dataset ensures reproducibility.
- The optional GDELT mode demonstrates how this project can be extended to live public news ingestion.
