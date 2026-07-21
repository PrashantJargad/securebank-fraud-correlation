"""
Export the synthetic training data to a CSV so you can see every row the model
learns from.

    python export_data.py     ->  writes training_data.csv

This is the EXACT data the model trains on. Because the generator uses a fixed
random seed, re-running it produces the identical dataset every time — so this
CSV is a faithful snapshot, not a one-off sample.
"""

import csv
import os

from ml import generate_dataset
from features import FEATURES

OUT = os.path.join(os.path.abspath(os.path.dirname(__file__)), "training_data.csv")


def export(path=OUT):
    X, y = generate_dataset()
    float4 = {"amount_fraction"}
    float2 = {"beneficiary_age_min", "amount"}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "label_name"] + FEATURES)
        for row, label in zip(X, y):
            vals = []
            for name, v in zip(FEATURES, row):
                if name in float4:
                    vals.append(round(float(v), 4))
                elif name in float2:
                    vals.append(round(float(v), 2))
                else:
                    vals.append(int(v))
            w.writerow([int(label), "fraud" if int(label) == 1 else "legit"] + vals)
    return path, len(X)


if __name__ == "__main__":
    path, n = export()
    print(f"Wrote {n} rows -> {path}")
