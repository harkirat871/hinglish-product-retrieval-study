# Reproducibility guide

## Environment

- Python 3.10 or later
- Packages listed in the root requirements.txt
- A LaTeX distribution with Biber for local manuscript compilation, or Overleaf

## Re-run the experiment

Run these commands from the repository root:

~~~bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python src/experiment.py
python src/make_figures.py
~~~

The experiment script initializes Python's random generator and NumPy with seed 42. It regenerates data/catalog.csv and data/queries.csv, evaluates the three retrieval methods, and writes results/results.json.

The figure script reads results/results.json and creates vector PDF and high-resolution PNG figures under paper/figures/ and results/figures/.

## Compile the manuscript

From the paper directory:

~~~bash
latexmk -pdf main.tex
~~~

The bibliography is managed with BibLaTeX/Biber and is stored in references.bib. The manuscript references the generated PDF figures in the figures directory.

## Evaluation details

- The query generator creates 60 records for each of three conditions: clean, light variation, and heavy variation.
- The supplied dataset contains 180 query records: 164 answerable and 16 unanswerable.
- The 16 unanswerable queries are excluded from Top-1 and Top-3 accuracy because no relevant catalog identifier exists for them.
- Top-1 confidence intervals are 95% percentile-bootstrap intervals based on 2,000 resamples of query-level binary correctness indicators.
- The current report treats the intervals as descriptive. It does not conduct a paired significance test.
- The 180 query rows contain 173 unique query strings. Repeated strings and shared templates may induce dependence; the simple query-level bootstrap does not model this dependence.
- The hybrid RRF implementation retains the original zero-based rank-position convention. The manuscript documents this implementation detail explicitly.

## Reproducibility caveats

Exact figure appearance may vary slightly with Matplotlib and font versions. Numerical results may vary with dependency versions or implementation changes. Record the Python and package versions used for any future rerun and do not silently replace the committed result file with newly generated results.
