"""
Quick local scorer: compares submission.csv against ground_truth.csv
and prints accuracy + F1.

Assumes:
- ground_truth.csv has columns: video_id, label
- submission.csv    has columns: video_id, predicted_label
- label values are "bully" / "no-bully"

Usage:
    python score_submission.py
Run it from the folder containing both csv files.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

GROUND_TRUTH_CSV = "ground_truth.csv"
SUBMISSION_CSV = "submission.csv"

gt = pd.read_csv(GROUND_TRUTH_CSV)
sub = pd.read_csv(SUBMISSION_CSV)

merged = gt.merge(sub, on="video_id", how="inner")

if len(merged) != len(gt):
    print(f"Warning: matched {len(merged)} rows, but ground_truth has {len(gt)}.")

acc = accuracy_score(merged["label"], merged["predicted_label"])
precision, recall, f1, _ = precision_recall_fscore_support(
    merged["label"], merged["predicted_label"], average="binary", pos_label="bully", zero_division=0
)

print(f"Videos scored: {len(merged)}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
