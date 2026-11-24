# Typo-Robust Product Retrieval for Hinglish E-Commerce Queries

A reproducible synthetic pilot study comparing BM25, character n-gram TF-IDF, and hybrid Reciprocal Rank Fusion (RRF) on Roman-script Hinglish-style automotive product queries with controlled spelling variation.

> **Dataset status:** The catalog and all queries are synthetic. They are not real customer messages, scraped retailer listings, verified inventory, or validated vehicle-fitment records. Results apply only to this constructed benchmark.

## Main result

On 164 answerable synthetic queries, character n-gram TF-IDF achieved 89.6% overall Top-1 accuracy (95% percentile-bootstrap interval: 84.8%–93.9%), compared with 72.6% for BM25 and 74.4% for hybrid RRF. Under heavy spelling variation, the methods achieved 80.7%, 26.3%, and 33.3%, respectively.

## Reproduce

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python src/experiment.py
python src/make_figures.py
```

The experiment script regenerates the synthetic CSV files and result JSON with fixed random seeds. The figure script creates PDF and PNG figures. To compile the LaTeX manuscript, install a LaTeX distribution with Biber and run `latexmk -pdf main.tex` from `paper/`, or upload the manuscript files to Overleaf.

## Methods

- **BM25:** word-token retrieval baseline.
- **Character TF-IDF:** character n-gram retrieval with the `char_wb` analyzer and n-gram range 3–5.
- **Hybrid RRF:** combines the two rankings using Reciprocal Rank Fusion.

Top-1 and Top-3 accuracy are calculated on answerable queries only. Sixteen unanswerable queries are excluded from these metrics.

## Scope

This is an exploratory, synthetic pilot study. It is not peer-reviewed. It evaluates retrieval ranking only—not LLM response generation, a chatbot, real product compatibility, live inventory, or customer outcomes. See `documentation/` for the data dictionary, reproducibility notes, and AI-assistance disclosure.
