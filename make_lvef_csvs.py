"""Create LVEF probe eval CSVs in EchoJEPA format.

Joins LVEF labels to AVI paths, splits train/val by study,
z-scores LVEF from train stats, writes space-delimited CSVs.

Usage:
    python make_lvef_csvs.py
"""

import os
import glob
import numpy as np
import pandas as pd

LVEF_CSV = "/lab-share/Cardio-Mayourian-e2/Public/EchoFocus_Features/Bill_Features/Subgroup_Details_from_ECGLVEF.csv"
AVI_DIR = "/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k"
OUT_DIR = "/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/data"
SEED = 42
VAL_FRAC = 0.2

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load LVEF labels
    df = pd.read_csv(LVEF_CSV)
    df["Event.ID.Number"] = df["Event.ID.Number"].astype(int).astype(str)
    print(f"LVEF rows: {len(df)}", flush=True)

    # Get directory names, strip _trim
    dirs = os.listdir(AVI_DIR)
    dir_ids = {d.replace("_trim", ""): d for d in dirs if os.path.isdir(os.path.join(AVI_DIR, d))}
    print(f"Directories: {len(dir_ids)}", flush=True)

    # Match
    df = df[df["Event.ID.Number"].isin(dir_ids)]
    df = df.dropna(subset=["LVEF"])
    df = df.drop_duplicates(subset=["Event.ID.Number"])
    print(f"Matched studies with LVEF: {len(df)}", flush=True)

    # Split by study (no patient leakage — one row per study after dedup)
    rng = np.random.RandomState(SEED)
    study_ids = df["Event.ID.Number"].values.copy()
    rng.shuffle(study_ids)
    n_val = int(len(study_ids) * VAL_FRAC)
    val_ids = set(study_ids[:n_val])
    train_ids = set(study_ids[n_val:])
    print(f"Train studies: {len(train_ids)}, Val studies: {len(val_ids)}", flush=True)

    # Z-score from train stats
    train_lvef = df[df["Event.ID.Number"].isin(train_ids)]["LVEF"].values
    lvef_mean = train_lvef.mean()
    lvef_std = train_lvef.std()
    print(f"LVEF mean: {lvef_mean:.4f}, std: {lvef_std:.4f}", flush=True)

    # Build rows: one AVI path + z-scored LVEF per line
    def build_rows(ids):
        rows = []
        for sid in ids:
            dirname = dir_ids[sid]
            lvef = df[df["Event.ID.Number"] == sid]["LVEF"].values[0]
            z = (lvef - lvef_mean) / lvef_std
            avis = glob.glob(os.path.join(AVI_DIR, dirname, "*.avi"))
            for avi in avis:
                rows.append(f"{avi} {z}")
        return rows

    print("Building train rows...", flush=True)
    train_rows = build_rows(train_ids)
    print(f"Train AVIs: {len(train_rows)}", flush=True)

    print("Building val rows...", flush=True)
    val_rows = build_rows(val_ids)
    print(f"Val AVIs: {len(val_rows)}", flush=True)

    # Write
    train_path = os.path.join(OUT_DIR, "lvef_train.csv")
    val_path = os.path.join(OUT_DIR, "lvef_val.csv")

    with open(train_path, "w") as f:
        f.write("\n".join(train_rows) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_rows) + "\n")

    # Save scaler info
    with open(os.path.join(OUT_DIR, "lvef_scaler.txt"), "w") as f:
        f.write(f"mean={lvef_mean}\nstd={lvef_std}\n")

    print(f"Wrote {train_path} ({len(train_rows)} rows)", flush=True)
    print(f"Wrote {val_path} ({len(val_rows)} rows)", flush=True)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
