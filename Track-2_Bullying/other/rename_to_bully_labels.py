"""
Run this once, locally, before you zip/upload the dataset to Kaggle.

What it does:
1. Renames RWF-2000/train/Fight     -> RWF-2000/train/bully
   Renames RWF-2000/train/NonFight  -> RWF-2000/train/no-bully
2. Rewrites ground_truth.csv so any "Fight"/"NonFight" or
   "violence"/"no violence"-style labels become "bully"/"no-bully".
   (It auto-detects which label column and which wording is used —
   check the printed preview at the end to make sure it mapped things
   correctly for YOUR csv's actual column names.)

Usage:
    python rename_to_bully_labels.py
Run it from inside Track-2_Bullying/, next to RWF-2000/ and ground_truth.csv.
"""

import os
import shutil
import pandas as pd

DATASET_ROOT = "RWF-2000"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
GROUND_TRUTH_CSV = "ground_truth.csv"

# Old name -> new name, checked case-insensitively
DIR_RENAME_MAP = {
    "fight": "bully",
    "nonfight": "no-bully",
}

# Any of these text values (case-insensitive, exact match) get mapped
LABEL_VALUE_MAP = {
    "fight": "bully",
    "nonfight": "no-bully",
    "non-fight": "no-bully",
    "violence": "bully",
    "novilence": "no-bully",
    "no violence": "no-bully",
    "no-violence": "no-bully",
    "nonviolence": "no-bully",
}


def rename_train_dirs():
    if not os.path.isdir(TRAIN_DIR):
        print(f"Skipping directory rename: '{TRAIN_DIR}' not found.")
        return
    for entry in os.listdir(TRAIN_DIR):
        full_path = os.path.join(TRAIN_DIR, entry)
        if not os.path.isdir(full_path):
            continue
        new_name = DIR_RENAME_MAP.get(entry.lower())
        if new_name:
            new_path = os.path.join(TRAIN_DIR, new_name)
            if os.path.exists(new_path):
                print(f"Skipping '{entry}' -> '{new_name}': target already exists.")
                continue
            shutil.move(full_path, new_path)
            print(f"Renamed folder: {entry} -> {new_name}")
        else:
            print(f"Left untouched (no mapping for): {entry}")


def fix_ground_truth_csv():
    if not os.path.isfile(GROUND_TRUTH_CSV):
        print(f"Skipping csv fix: '{GROUND_TRUTH_CSV}' not found.")
        return

    df = pd.read_csv(GROUND_TRUTH_CSV)
    print("\nOriginal ground_truth.csv preview:")
    print(df.head())

    changed_any = False
    for col in df.columns:
        if df[col].dtype == object:
            lowered = df[col].astype(str).str.lower().str.strip()
            if lowered.isin(LABEL_VALUE_MAP.keys()).any():
                df[col] = lowered.map(LABEL_VALUE_MAP).fillna(df[col])
                changed_any = True
                print(f"\nUpdated column '{col}' using the label map.")

    if changed_any:
        backup_path = GROUND_TRUTH_CSV.replace(".csv", "_backup.csv")
        shutil.copy(GROUND_TRUTH_CSV, backup_path)
        df.to_csv(GROUND_TRUTH_CSV, index=False)
        print(f"\nSaved updated {GROUND_TRUTH_CSV} (original backed up to {backup_path}).")
        print("New preview:")
        print(df.head())
    else:
        print("\nNo matching label text found in ground_truth.csv — nothing changed.")
        print("Open the csv and check LABEL_VALUE_MAP above matches your actual label text.")


if __name__ == "__main__":
    rename_train_dirs()
    fix_ground_truth_csv()
    print("\nDone. train/ should now contain 'bully' and 'no-bully' folders.")
