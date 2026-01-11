import pandas as pd
from sklearn.impute import SimpleImputer

# ==============================================================
# STEP 1: Load datasets
# ==============================================================

print("🔹 Loading datasets...")

training_matrix = pd.read_csv("data/raw/training_matrix.csv")
moca_df = pd.read_csv(
    "data/raw/Montreal_Cognitive_Assessment__MoCA__05Sep2025.csv")
rbd_df = pd.read_csv(
    "data/raw/REM_Sleep_Behavior_Disorder_Questionnaire_05Sep2025.csv")
upsit_df = pd.read_csv(
    "data/raw/University_of_Pennsylvania_Smell_Identification_Test_UPSIT_05Sep2025.csv")

print("✅ Datasets loaded")
print(f"Training matrix shape: {training_matrix.shape}")
print(f"MoCA shape: {moca_df.shape}")
print(f"RBD shape: {rbd_df.shape}")
print(f"UPSIT shape: {upsit_df.shape}\n")


# ==============================================================
# STEP 2: Standardize labels
# ==============================================================

print("🔹 Standardizing labels...")

if "COHORT" in training_matrix.columns:
    training_matrix["LABEL"] = training_matrix["COHORT"].apply(
        lambda x: 1 if x == "PD" else 0)
    print("✅ Added LABEL column from COHORT")
    print(training_matrix["LABEL"].value_counts())
else:
    print("⚠️ No COHORT column found. You will need diagnosis/conversion data to create labels.\n")


# ==============================================================
# STEP 3: Convert EVENT_ID → Years
# ==============================================================

print("🔹 Converting EVENT_ID to YEARS...")


def event_to_year(event_id):
    if pd.isna(event_id):
        return None
    if event_id == "BL":
        return 0
    try:
        return int(event_id[1:])  # V01 -> 1, V02 -> 2, etc.
    except:
        return None


for df in [moca_df, rbd_df, upsit_df]:
    if "EVENT_ID" in df.columns:
        df["YEARS"] = df["EVENT_ID"].apply(event_to_year)

print("✅ EVENT_ID converted. Sample:")
print(moca_df[["PATNO", "EVENT_ID", "YEARS"]].head(), "\n")


# ==============================================================
# STEP 4: Feature engineering
# ==============================================================

print("🔹 Feature engineering...")

# --- MoCA ---
if "MCATOT" in moca_df.columns:
    moca_features = moca_df.groupby("PATNO").agg(
        MCATOT_mean=("MCATOT", "mean"),
        MCATOT_slope=(
            "MCATOT", lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
    ).reset_index()
    print("✅ MoCA features created:", moca_features.shape)
else:
    moca_features = pd.DataFrame()
    print("⚠️ MoCA total score column not found")

# --- RBD ---
print("RBD columns (first 20):", rbd_df.columns[:20])
if "RBDTOTS" in rbd_df.columns:
    rbd_df["RBD_Positive"] = rbd_df["RBDTOTS"].apply(
        lambda x: 1 if x > 4 else 0)
    rbd_features = rbd_df.groupby("PATNO").agg(
        RBD_mean=("RBDTOTS", "mean"),
        RBD_Positive=("RBD_Positive", "max")
    ).reset_index()
    print("✅ RBD features created:", rbd_features.shape)
else:
    rbd_features = pd.DataFrame()
    print("⚠️ RBD total score column not found")

# --- UPSIT ---
print("UPSIT columns (first 20):", upsit_df.columns[:20])
if "UPSITOT" in upsit_df.columns:
    upsit_df["UPSIT_anosmia"] = upsit_df["UPSITOT"].apply(
        lambda x: 1 if x < 18 else 0)
    upsit_features = upsit_df.groupby("PATNO").agg(
        UPSIT_mean=("UPSITOT", "mean"),
        UPSIT_anosmia=("UPSIT_anosmia", "max")
    ).reset_index()
    print("✅ UPSIT features created:", upsit_features.shape)
else:
    upsit_features = pd.DataFrame()
    print("⚠️ UPSIT total score column not found")

print()


# ==============================================================
# STEP 5: Standardize ID columns & Merge
# ==============================================================

print("🔹 Standardizing ID columns...")

# Rename participant ID columns to common name: PATNO
for df in [training_matrix, moca_features, rbd_features, upsit_features]:
    if not df.empty:
        for col in df.columns:
            if col.lower() in ["patno", "patient_id", "subject_id", "id", "rid"]:
                df.rename(columns={col: "PATNO"}, inplace=True)

print("✅ Standardized ID columns")
print("Training matrix ID col present?",
      "PATNO" in training_matrix.columns, "\n")

# Merge
print("🔹 Merging datasets...")
final_df = training_matrix.copy()

for feat_df in [moca_features, rbd_features, upsit_features]:
    if not feat_df.empty:
        final_df = final_df.merge(feat_df, on="PATNO", how="left")

print("✅ Merged final dataset")
print("Shape:", final_df.shape)
print(final_df.head(), "\n")


# ==============================================================
# STEP 6: Handle missing values
# ==============================================================

print("🔹 Handling missing values...")

feature_cols = ["MCATOT_mean", "MCATOT_slope", "RBD_mean",
                "RBD_Positive", "UPSIT_mean", "UPSIT_anosmia"]
available_features = [col for col in feature_cols if col in final_df.columns]

if available_features:
    imputer = SimpleImputer(strategy="median")
    final_df[available_features] = imputer.fit_transform(
        final_df[available_features])
    print("✅ Missing values imputed with median")
    print(final_df[available_features].isnull().sum(), "\n")
else:
    print("⚠️ No engineered features found to impute.\n")


# ==============================================================
# STEP 7: Save final dataset
# ==============================================================

out_path = "data/processed/final_dataset.csv"
final_df.to_csv(out_path, index=False)

print(f"✅ Final dataset saved at {out_path}")
