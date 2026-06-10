#!/usr/bin/env python3
"""
Automated Sentiment Analysis Pipeline for News Articles

Features
--------
1. Loads a bundled 100-article renewable-energy news dataset, or optionally collects
   articles through the public GDELT DOC API.
2. Applies NLP preprocessing with NLTK: tokenization, stop-word removal, and lemmatization.
3. Scores article sentiment with VADER when available and a small fallback lexicon otherwise.
4. Saves article-level sentiment outputs to CSV.
5. Generates a Markdown report with sentiment distribution and top positive/negative keywords.

Example
-------
python src/news_sentiment_pipeline.py \
  --input data/renewable_energy_news_100.csv \
  --output outputs/sentiment_results.csv \
  --report reports/sentiment_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover - fallback keeps script runnable in constrained envs
    nltk = None
    stopwords = None
    SentimentIntensityAnalyzer = None
    WordNetLemmatizer = None
    word_tokenize = None


DEFAULT_INPUT = Path("data/renewable_energy_news_100.csv")
DEFAULT_OUTPUT = Path("outputs/sentiment_results.csv")
DEFAULT_REPORT = Path("reports/sentiment_report.md")
DEFAULT_TOPIC = "renewable energy"

POSITIVE_TERMS = {
    "strong", "growth", "gain", "gains", "improved", "improve", "positive", "profitable",
    "profitability", "support", "favorable", "resilient", "reliability", "create", "jobs",
    "encouraging", "secured", "expanded", "award", "praised", "stabilize", "investment",
}
NEGATIVE_TERMS = {
    "risk", "risks", "delay", "delays", "shortages", "pressure", "concern", "concerns",
    "criticized", "criticism", "missed", "disappointing", "cut", "paused", "inflation",
    "bottlenecks", "uncertainty", "weaker", "cautiously", "costs", "slow",
}


@dataclass
class NLPResources:
    stop_words: set
    lemmatizer: Optional[object]
    vader: Optional[object]


def ensure_nltk_resources() -> NLPResources:
    """Load required NLTK resources. Download when possible; gracefully fall back otherwise."""
    if nltk is None:
        return NLPResources(stop_words=basic_stop_words(), lemmatizer=None, vader=None)

    required = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ]
    for lookup_path, package_name in required:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                pass

    try:
        sw = set(stopwords.words("english")) if stopwords else basic_stop_words()
    except Exception:
        sw = basic_stop_words()

    try:
        lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None
    except Exception:
        lemmatizer = None

    try:
        vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
    except Exception:
        vader = None

    return NLPResources(stop_words=sw, lemmatizer=lemmatizer, vader=vader)


def basic_stop_words() -> set:
    return {
        "a", "an", "the", "and", "or", "but", "if", "while", "of", "to", "in", "on", "for",
        "with", "without", "by", "from", "as", "at", "is", "are", "was", "were", "be", "been",
        "being", "this", "that", "these", "those", "it", "its", "into", "over", "under", "could",
        "would", "should", "may", "might", "will", "said", "also", "about", "around", "after",
        "before", "during", "through", "than", "then", "their", "them", "they", "he", "she",
    }


def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    if word_tokenize:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text)


def preprocess_text(text: str, resources: NLPResources) -> Tuple[str, List[str]]:
    """Tokenize, remove stop words, and lemmatize article text."""
    tokens = tokenize_text(text.lower())
    clean_tokens: List[str] = []

    for token in tokens:
        token = re.sub(r"[^a-z']", "", token)
        if not token or len(token) < 3 or token in resources.stop_words:
            continue
        if resources.lemmatizer:
            try:
                token = resources.lemmatizer.lemmatize(token)
            except Exception:
                pass
        clean_tokens.append(token)

    return " ".join(clean_tokens), clean_tokens


def fallback_sentiment(tokens: List[str]) -> Dict[str, float]:
    pos = sum(1 for t in tokens if t in POSITIVE_TERMS)
    neg = sum(1 for t in tokens if t in NEGATIVE_TERMS)
    total = max(pos + neg, 1)
    compound = (pos - neg) / total
    return {
        "neg": neg / total,
        "neu": max(0.0, 1.0 - ((pos + neg) / max(len(tokens), 1))),
        "pos": pos / total,
        "compound": compound,
    }


def score_sentiment(original_text: str, tokens: List[str], resources: NLPResources) -> Dict[str, float]:
    if resources.vader:
        return resources.vader.polarity_scores(original_text or "")
    return fallback_sentiment(tokens)


def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def collect_gdelt_articles(topic: str, limit: int, output_path: Path) -> pd.DataFrame:
    """Collect article metadata from the public GDELT DOC 2.0 API and attempt light text extraction.

    GDELT returns article URLs and metadata, not guaranteed full text. This function attempts to
    fetch accessible article pages and extract paragraph text. It is optional because news sites
    may block scraping or change layouts.
    """
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    end = datetime.utcnow()
    start = end - timedelta(days=90)
    params = {
        "query": topic,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max(limit * 2, limit), 250),
        "sort": "HybridRel",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    articles = response.json().get("articles", [])

    records = []
    for idx, item in enumerate(articles, start=1):
        if len(records) >= limit:
            break
        url = item.get("url", "")
        title = item.get("title", "")
        text = extract_article_text(url)
        if not text:
            text = title
        if len(text.split()) < 20:
            continue
        records.append({
            "article_id": f"GDELT_{len(records)+1:03d}",
            "topic": topic,
            "source": item.get("domain", "GDELT"),
            "source_url": url,
            "title": title,
            "published_date": item.get("seendate", ""),
            "article_text": text,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def extract_article_text(url: str) -> str:
    if not url:
        return ""
    try:
        headers = {"User-Agent": "news-sentiment-portfolio-project/1.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paragraphs if len(p.split()) >= 5)
        return re.sub(r"\s+", " ", text)[:12000]
    except Exception:
        return ""


def validate_articles(df: pd.DataFrame) -> None:
    required = {"article_id", "title", "published_date", "article_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {sorted(missing)}")
    if len(df) < 100:
        raise ValueError(f"Expected at least 100 articles; found {len(df)}")


def analyze_articles(df: pd.DataFrame, resources: NLPResources) -> Tuple[pd.DataFrame, Dict[str, object]]:
    validate_articles(df)
    rows = []
    positive_keyword_counter: Counter = Counter()
    negative_keyword_counter: Counter = Counter()
    overall_counter: Counter = Counter()

    for _, row in df.iterrows():
        original_text = str(row.get("article_text", ""))
        processed_text, tokens = preprocess_text(original_text, resources)
        scores = score_sentiment(original_text, tokens, resources)
        label = sentiment_label(float(scores.get("compound", 0)))

        overall_counter.update(tokens)
        if label == "positive":
            positive_keyword_counter.update(tokens)
        elif label == "negative":
            negative_keyword_counter.update(tokens)

        rows.append({
            "article_id": row.get("article_id"),
            "topic": row.get("topic", DEFAULT_TOPIC),
            "source": row.get("source", ""),
            "source_url": row.get("source_url", ""),
            "title": row.get("title", ""),
            "published_date": row.get("published_date", ""),
            "article_text": original_text,
            "processed_text": processed_text,
            "token_count": len(tokens),
            "sentiment_negative": round(float(scores.get("neg", 0)), 4),
            "sentiment_neutral": round(float(scores.get("neu", 0)), 4),
            "sentiment_positive": round(float(scores.get("pos", 0)), 4),
            "sentiment_compound": round(float(scores.get("compound", 0)), 4),
            "sentiment_label": label,
        })

    results = pd.DataFrame(rows)
    summary = {
        "article_count": int(len(results)),
        "sentiment_distribution": results["sentiment_label"].value_counts().to_dict(),
        "average_compound_score": round(float(results["sentiment_compound"].mean()), 4),
        "top_positive_keywords": positive_keyword_counter.most_common(15),
        "top_negative_keywords": negative_keyword_counter.most_common(15),
        "top_overall_keywords": overall_counter.most_common(20),
    }
    return results, summary


def write_report(summary: Dict[str, object], output_csv: Path, report_path: Path, topic: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dist = summary["sentiment_distribution"]
    total = summary["article_count"]

    def pct(count: int) -> str:
        return f"{(count / total * 100):.1f}%" if total else "0.0%"

    lines = [
        "# Sentiment Analysis Report",
        "",
        f"**Topic:** {topic}",
        f"**Articles processed:** {total}",
        f"**Output dataset:** `{output_csv.as_posix()}`",
        f"**Average compound sentiment score:** {summary['average_compound_score']}",
        "",
        "## Methodology",
        "",
        "The pipeline loads news articles, applies NLTK-based preprocessing, and scores sentiment using VADER when available. Preprocessing includes tokenization, lowercasing, punctuation removal, stop-word removal, and lemmatization. The article-level output preserves the original article text, processed text, VADER sentiment scores, and a final positive/neutral/negative label.",
        "",
        "## Sentiment Distribution",
        "",
        "| Sentiment | Count | Share |",
        "|---|---:|---:|",
    ]
    for label in ["positive", "neutral", "negative"]:
        count = int(dist.get(label, 0))
        lines.append(f"| {label.title()} | {count} | {pct(count)} |")

    lines.extend([
        "",
        "## Top Keywords in Positive Articles",
        "",
        "| Keyword | Frequency |",
        "|---|---:|",
    ])
    for term, count in summary["top_positive_keywords"]:
        lines.append(f"| {term} | {count} |")

    lines.extend([
        "",
        "## Top Keywords in Negative Articles",
        "",
        "| Keyword | Frequency |",
        "|---|---:|",
    ])
    for term, count in summary["top_negative_keywords"]:
        lines.append(f"| {term} | {count} |")

    lines.extend([
        "",
        "## Top Overall Keywords",
        "",
        "| Keyword | Frequency |",
        "|---|---:|",
    ])
    for term, count in summary["top_overall_keywords"]:
        lines.append(f"| {term} | {count} |")

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- `sentiment_compound` ranges from -1 to +1.",
        "- Scores >= 0.05 are labeled positive.",
        "- Scores <= -0.05 are labeled negative.",
        "- Scores between -0.05 and 0.05 are labeled neutral.",
        "- Keyword tables are based on cleaned and lemmatized tokens after stop-word removal.",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated sentiment analysis pipeline for news articles")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input article CSV with at least 100 rows")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown report path")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="News topic for reporting and optional collection")
    parser.add_argument("--collect-live", action="store_true", help="Collect articles from GDELT before analysis")
    parser.add_argument("--limit", type=int, default=100, help="Number of articles to collect in live mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resources = ensure_nltk_resources()

    if args.collect_live:
        print(f"Collecting up to {args.limit} live articles from GDELT for topic: {args.topic}")
        df = collect_gdelt_articles(args.topic, args.limit, args.input)
    else:
        df = pd.read_csv(args.input)

    results, summary = analyze_articles(df, resources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    write_report(summary, args.output, args.report, args.topic)

    print("Sentiment pipeline completed successfully.")
    print(f"Articles processed: {summary['article_count']}")
    print(f"Sentiment distribution: {summary['sentiment_distribution']}")
    print(f"Output CSV: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
