# Summary Quality Evaluator

Python batch evaluator for comparing source documents against summaries or categorized outputs. It combines semantic similarity, learned quality scoring, and entailment checks into a CSV report for review workflows.

## What It Measures

- BERTScore for semantic similarity between source text and summary.
- BLEURT for learned summary quality and factuality scoring.
- RoBERTa MNLI entailment probability for logical consistency.
- Overall average score across the three metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas tqdm bert-score bleurt transformers torch
```

BLEURT model files are expected to be available locally as `BLEURT-20`.

## Usage

Folder mode expects matching filenames in `full_texts/` and `summaries/`:

```text
data/
  full_texts/
    doc1.txt
  summaries/
    doc1.txt
```

Run:

```bash
python evalue_summary_quality.py --folder data/ --output summary_eval_results.csv
```

CSV mode expects columns named `full_text` and `summary`:

```bash
python evalue_summary_quality.py --csv pairs.csv --output summary_eval_results.csv
```

## Output

The script writes a CSV with per-document `bert_score`, `bleurt`, `nli`, and `overall` fields.
