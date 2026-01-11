# src/inventory.py
import os
import re
import pandas as pd
from glob import glob

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob(os.path.join(RAW, "*")))
summary = []


def sample_df(path, nrows=200):
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path, nrows=nrows, dtype=str, keep_default_na=False, na_values=[""])
        elif path.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(path, nrows=nrows, dtype=str)
    except Exception as e:
        return None


pat_re = re.compile(r"pat|subject|subj|id", re.I)
visit_re = re.compile(r"visit|event|vst|assessment|timepoint", re.I)
biomarker_re = re.compile(
    r"alpha|synuclein|syn|tau|ptau|p-tau|abeta|aβ|abeta42|abeta40|nfl|neurofil|neurofilament", re.I)

for f in files:
    info = {"file": os.path.basename(f), "path": f}
    df = sample_df(f)
    if df is None:
        info.update(
            {"error": "cannot read (maybe binary/non-tabular)", "sample_cols": None})
        summary.append(info)
        continue
    cols = list(df.columns)
    info["sample_rows"] = len(df)
    info["n_cols"] = len(cols)
    info["sample_cols"] = cols[:30]
    # find probable id/visit/biomarker cols in the sample
    pat_cols = [c for c in cols if pat_re.search(c)]
    visit_cols = [c for c in cols if visit_re.search(c)]
    biomarker_cols = [c for c in cols if biomarker_re.search(c)]
    info.update({
        "probable_patno_cols": pat_cols,
        "probable_visit_cols": visit_cols,
        "probable_biomarker_cols_sample": biomarker_cols,
    })
    # count unique PATNO if found in sample
    if pat_cols:
        try:
            pc = pat_cols[0]
            info["unique_patno_in_sample"] = df[pc].nunique()
            info["sample_patno_examples"] = list(
                pd.Series(df[pc].dropna().unique())[:5])
        except Exception:
            info["unique_patno_in_sample"] = None
    else:
        info["unique_patno_in_sample"] = None
    # missingness (sample)
    miss = df.isna().mean().sort_values(ascending=False).head(10).to_dict()
    info["top_missingness_sample"] = miss
    summary.append(info)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT, "dataset_inventory.csv"), index=False)
print("Inventory written to:", os.path.join(OUT, "dataset_inventory.csv"))
