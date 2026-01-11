import pandas as pd

# -------------------------------
# 1️⃣ Load CSV safely
# -------------------------------
df = pd.read_csv(
    "data/processed/cleaned_Current_Biospecimen_Analysis_Results_05Sep2025.csv",
    dtype=str,        # treat all values as string initially
    low_memory=False  # avoid dtype warnings
)

# Strip any hidden spaces from column names
df.columns = df.columns.str.strip()

# -------------------------------
# 2️⃣ Keep only relevant columns
# -------------------------------
required_cols = ["patno", "event_id", "TESTNAME", "TESTVALUE"]
df = df[required_cols].copy()

# -------------------------------
# 3️⃣ Optional: remove exact duplicates
# -------------------------------
df = df.drop_duplicates(subset=required_cols)

# -------------------------------
# 4️⃣ Pivot table
# -------------------------------
df_wide = df.pivot_table(
    index=["patno", "event_id"],
    columns="TESTNAME",
    values="TESTVALUE",
    aggfunc="first"  # take first in case of duplicates
).reset_index()

# Flatten column multi-index
df_wide.columns.name = None

# -------------------------------
# 5️⃣ Optional: convert numeric columns
# -------------------------------
# Only convert columns that are not patno/event_id
for col in df_wide.columns:
    if col not in ["patno", "event_id"]:
        df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce")

# -------------------------------
# 6️⃣ Save to CSV
# -------------------------------
outpath = "data/processed/cleaned_csf_biospecimen_wide.csv"
df_wide.to_csv(outpath, index=False)

# -------------------------------
# 7️⃣ Summary
# -------------------------------
print(f"✅ Pivoted biospecimen saved to {outpath}")
print("Shape:", df_wide.shape)
print("Columns (first 20):", df_wide.columns.tolist()[:20])
