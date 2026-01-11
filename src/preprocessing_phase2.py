import pandas as pd
import numpy as np
import os

# ============================
# Load the already merged dataset
# ============================
print("🔹 Loading base dataset (final_dataset.csv)...")
base_path = "data/processed/final_dataset.csv"
df = pd.read_csv(base_path)
print(f"✅ Base dataset loaded: {df.shape}")

# ============================
# Load raw RBD + UPSIT datasets again
# ============================
rbd_path = "data/raw/REM_Sleep_Behavior_Disorder_Questionnaire_05Sep2025.csv"
upsit_path = "data/raw/University_of_Pennsylvania_Smell_Identification_Test_UPSIT_05Sep2025.csv"

rbd = pd.read_csv(rbd_path)
upsit = pd.read_csv(upsit_path)

print(f"RBD raw: {rbd.shape}")
print(f"UPSIT raw: {upsit.shape}")

# ============================
# Feature engineering: RBD
# ============================
print("🔹 Engineering RBD features...")

rbd_cols = [
    "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL",
    "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB"
]
rbd["RBD_total"] = rbd[rbd_cols].apply(
    pd.to_numeric, errors="coerce").sum(axis=1)

rbd_feat = rbd.groupby("PATNO").agg(
    RBD_mean=("RBD_total", "mean"),
    RBD_slope=("RBD_total", lambda x: np.polyfit(
        range(len(x)), x, 1)[0] if len(x) > 1 else 0)
).reset_index()

print(f"✅ RBD features created: {rbd_feat.shape}")

# ============================
# Feature engineering: UPSIT
# ============================
print("🔹 Engineering UPSIT features...")

upsit_correct_cols = [c for c in upsit.columns if c.startswith(
    "SCENT_") and c.endswith("_CORRECT")]
upsit["UPSIT_total"] = upsit[upsit_correct_cols].apply(
    pd.to_numeric, errors="coerce").sum(axis=1)

upsit_feat = upsit.groupby("PATNO").agg(
    UPSIT_mean=("UPSIT_total", "mean"),
    UPSIT_slope=("UPSIT_total", lambda x: np.polyfit(
        range(len(x)), x, 1)[0] if len(x) > 1 else 0)
).reset_index()

print(f"✅ UPSIT features created: {upsit_feat.shape}")

# ============================
# Merge back into dataset
# ============================
print("🔹 Merging enhanced features into dataset...")

enhanced = df.merge(rbd_feat, on="PATNO", how="left")
enhanced = enhanced.merge(upsit_feat, on="PATNO", how="left")

print(f"✅ Enhanced dataset shape: {enhanced.shape}")

# ============================
# Handle missing values (median imputation)
# ============================
print("🔹 Handling missing values for new features...")
for col in ["RBD_mean", "RBD_slope", "UPSIT_mean", "UPSIT_slope"]:
    if col in enhanced.columns:
        enhanced[col] = enhanced[col].fillna(enhanced[col].median())

print("✅ Missing values handled")

# ============================
# Add binary cutoff features
# ============================
print("🔹 Adding binary cutoff features...")

enhanced["RBD_present"] = (enhanced["RBD_mean"] > 0).astype(int)
enhanced["UPSIT_anosmia"] = (enhanced["UPSIT_mean"] < 18).astype(int)

print("✅ Binary features added: RBD_present, UPSIT_anosmia")
print(enhanced[["PATNO", "RBD_mean", "RBD_present",
      "UPSIT_mean", "UPSIT_anosmia"]].head())

# ============================
# Save final enhanced dataset
# ============================
out_path = "data/processed/final_dataset_enhanced.csv"
enhanced.to_csv(out_path, index=False)
print(f"✅ Enhanced dataset with binary features saved at {out_path}")
