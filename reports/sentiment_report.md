# Sentiment Analysis Report

**Topic:** renewable energy
**Articles processed:** 100
**Output dataset:** `outputs/sentiment_results.csv`
**Average compound sentiment score:** 0.136

## Methodology

The pipeline loads news articles, applies NLTK-based preprocessing, and scores sentiment using VADER when available. Preprocessing includes tokenization, lowercasing, punctuation removal, stop-word removal, and lemmatization. The article-level output preserves the original article text, processed text, VADER sentiment scores, and a final positive/neutral/negative label.

## Sentiment Distribution

| Sentiment | Count | Share |
|---|---:|---:|
| Positive | 63 | 63.0% |
| Neutral | 0 | 0.0% |
| Negative | 37 | 37.0% |

## Top Keywords in Positive Articles

| Keyword | Frequency |
|---|---:|
| energy | 219 |
| renewable | 202 |
| policy | 130 |
| solar | 81 |
| demand | 81 |
| grid | 79 |
| capacity | 78 |
| wind | 78 |
| storage | 76 |
| executives | 74 |
| reliability | 74 |
| market | 70 |
| update | 67 |
| incentives | 67 |
| sector | 63 |

## Top Keywords in Negative Articles

| Keyword | Frequency |
|---|---:|
| energy | 129 |
| renewable | 119 |
| policy | 74 |
| project | 54 |
| wind | 47 |
| risk | 47 |
| permitting | 45 |
| pressure | 45 |
| update | 44 |
| costs | 44 |
| solar | 44 |
| sector | 37 |
| continues | 37 |
| balance | 37 |
| incentives | 37 |

## Top Overall Keywords

| Keyword | Frequency |
|---|---:|
| energy | 348 |
| renewable | 321 |
| policy | 204 |
| solar | 125 |
| wind | 125 |
| demand | 118 |
| grid | 116 |
| capacity | 115 |
| storage | 113 |
| update | 111 |
| reliability | 111 |
| executives | 111 |
| permitting | 108 |
| costs | 107 |
| market | 107 |
| incentives | 104 |
| sector | 100 |
| continues | 100 |
| balance | 100 |
| technology | 100 |

## Interpretation Guide

- `sentiment_compound` ranges from -1 to +1.
- Scores >= 0.05 are labeled positive.
- Scores <= -0.05 are labeled negative.
- Scores between -0.05 and 0.05 are labeled neutral.
- Keyword tables are based on cleaned and lemmatized tokens after stop-word removal.