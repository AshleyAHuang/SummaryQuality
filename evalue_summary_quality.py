import os
import pandas as pd
from tqdm import tqdm
from bert_score import score as bert_score
from bleurt import score as bleurt_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

'''Compares full text files with their summarized or categorized versions and produces:
BERTScore (semantic similarity)
BLEURT (fluency and factual accuracy)
NLI entailment (logical consistency)
Overall average score
Outputs results into a CSV file.

Run these commands once:
pip install bert-score bleurt transformer

Organize your files like this:
data/
 ├─ full_texts/
 │   ├─ doc1.txt
 │   ├─ doc2.txt
 └─ summaries/
     ├─ doc1.txt
     ├─ doc2.txt

Create a CSV (e.g. pairs.csv) with absolute or relative file paths:
full_text,summary
full_texts/doc1.txt,summaries/doc1.txt
full_texts/doc2.txt,summaries/doc2.txt

Option 1 — Folder mode
python batch_summary_evaluator.py --folder data/
Option 2 — CSV mode
python batch_summary_evaluator.py --csv pairs.csv
You can also rename the output file:
python batch_summary_evaluator.py --folder data/ --output eval_results.csv

Output:
bert_score: measures semantic similarity between full text and summary.
bleurt: assesses overall quality and factual correctness.
nli: how likely the summary logically follows from the text (entailment).
overall: mean of all three metrics.
'''

print("Loading models...")
bleurt_model = bleurt_score.BleurtScorer("BLEURT-20")
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
print("Models loaded.\n")

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def compute_bert(ref, hyp):
    P, R, F1 = bert_score([hyp], [ref], lang='en', rescale_with_baseline=True)
    return F1[0].item()

def compute_bleurt(ref, hyp):
    return bleurt_model.score(references=[ref], candidates=[hyp])[0]

def compute_nli(ref, hyp):
    inputs = nli_tokenizer(hyp, ref, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).flatten()
    return probs[2].item()  # entailment probability

def evaluate_pair(full_text, summary):
    bert = compute_bert(full_text, summary)
    bleurt = compute_bleurt(full_text, summary)
    nli = compute_nli(full_text, summary)
    overall = (bert + bleurt + nli) / 3
    return bert, bleurt, nli, overall

def batch_evaluate(input_folder=None, csv_file=None, output_csv="summary_eval_results.csv"):
    results = []

    if csv_file:
        df = pd.read_csv(csv_file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating from CSV"):
            ref = read_file(row["full_text"])
            hyp = read_file(row["summary"])
            bert, bleurt, nli, overall = evaluate_pair(ref, hyp)
            results.append({
                "full_text": row["full_text"],
                "summary": row["summary"],
                "bert_score": bert,
                "bleurt": bleurt,
                "nli": nli,
                "overall": overall
            })

    elif input_folder:
        full_dir = os.path.join(input_folder, "full_texts")
        sum_dir = os.path.join(input_folder, "summaries")

        for filename in tqdm(os.listdir(full_dir), desc="Evaluating folder pairs"):
            full_path = os.path.join(full_dir, filename)
            sum_path = os.path.join(sum_dir, filename)
            if not os.path.exists(sum_path):
                print(f"Skipping {filename}: summary file missing.")
                continue
            ref = read_file(full_path)
            hyp = read_file(sum_path)
            bert, bleurt, nli, overall = evaluate_pair(ref, hyp)
            results.append({
                "filename": filename,
                "bert_score": bert,
                "bleurt": bleurt,
                "nli": nli,
                "overall": overall
            })

    else:
        raise ValueError("You must provide either input_folder or csv_file")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    return out_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch evaluate text summaries using BERTScore, BLEURT, and NLI.")
    parser.add_argument("--folder", help="Path to folder containing full_texts/ and summaries/")
    parser.add_argument("--csv", help="Optional CSV file with columns full_text and summary")
    parser.add_argument("--output", default="summary_eval_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    batch_evaluate(input_folder=args.folder, csv_file=args.csv, output_csv=args.output)
