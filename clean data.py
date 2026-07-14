"""
clean data.py
─────────────────────────────────────────────────────────────────────────────
Real Canine Data Cleaning, Validation, Processing & Model Evaluation
Thesis: IoT Monitoring of Vital Signs and Active Cooling for
        Canine Heat Stroke Prevention Using Binary Classification

State encoding from veryfinalfinalistcode.ino:
  0 = HEAT_STRESSED    (Peltier + Fans ON)
  1 = COOLDOWN_PELTIER (Peltier + Fans ON, hold phase)
  2 = COOLDOWN_FAN     (Peltier OFF, Fans ON)
  3 = NORMAL           (All cooling OFF)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Google Sheets Config
SHEET_ID = "1Fc4PQagPnOMZhwk4XmsJaKuyViKE2N2FE92HhPGMERY"
GID      = "272419857"  # The specific tab containing Rocky's data
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/1Fc4PQagPnOMZhwk4XmsJaKuyViKE2N2FE92HhPGMERY/export?format=csv&gid=272419857"

# Row slice of Rocky's real data within the CSV (0-based pandas index, after header row)
# Sheet row 3 = pandas index 1 (row 1 is the header, row 2 is the 'rocky' label row)
# Sheet row 17 = pandas index 16
DATA_ROW_START = 1
DATA_ROW_END   = 16   # exclusive (reads indexes 1 to 15, total 15 data rows)

# ── Body temperature correction parameters ────────────────────────────────────
CORE_OFFSET       = 9.0   # °C fixed core-surface gradient
CONTACT_K         = 0.25  # amplification of contact delta
MIN_CONTACT_DELTA = 0.0   # Allows processing of very close raw/env readings
BODY_TEMP_MIN     = 37.0  # °C physiological floor for dogs
BODY_TEMP_MAX     = 42.5  # °C physiological ceiling for dogs

# ── Threshold table (Table 2 — Revised) ──────────────────────────────────────
BODY_TEMP_THRESHOLD = 39.5   # > this -> stressed
RESP_RATE_THRESHOLD = 200.0  # >= this -> stressed (panting zone, sensor composite)
ENV_TEMP_THRESHOLD  = 30.0   # > this -> stressed

# ── Respiratory rate physiological bounds ─────────────────────────────────────
RESP_MIN = 50    # bpm — sensor normal lower bound
RESP_MAX = 400   # bpm — maximum panting rate

# ── State mapping (from veryfinalfinalistcode.ino CoolingState enum) ──────────
STATE_LABELS = {
    0: "HEAT_STRESSED",
    1: "COOLDOWN_PELTIER",
    2: "COOLDOWN_FAN",
    3: "NORMAL"
}


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD RAW DATA FROM GOOGLE SHEETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data from Google Sheets...")
print(f"URL: {SHEET_CSV_URL}")
print("=" * 60)

df_raw = pd.read_csv(SHEET_CSV_URL)

# Aggressive column header cleanup: lowercase, strip spaces, remove internal double spaces
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

print(f"Full sheet shape   : {df_raw.shape}")
print(f"Cleaned columns found: {list(df_raw.columns)}")

# Slice Rocky's canine data rows
df = df_raw.iloc[DATA_ROW_START:DATA_ROW_END].copy().reset_index(drop=True)
print(f"Sliced rows        : {len(df)} rows ({DATA_ROW_START} to {DATA_ROW_END})")

# Expanded robust mapping dictionary (supports both spaced and snake_case headers)
rename_dict = {
    'body temp'     : 'raw_body_temp',
    'body_temp'     : 'raw_body_temp',
    'respi'         : 'raw_respi',
    'resp_rate'     : 'raw_respi',
    'env temp'      : 'env_temp',
    'ambient_temp'  : 'env_temp',
    'ml (1=on)'     : 'system_pred',
    'ml_prediction' : 'system_pred',
    'state'         : 'raw_state',
    'cooling_state' : 'raw_state',
    'a'             : 'timestamp',
    'timestamp'     : 'timestamp'
}
df = df.rename(columns=rename_dict)

# Ensure all required processing columns exist to avoid KeyErrors
required_cols = ['raw_body_temp', 'raw_respi', 'env_temp', 'raw_state', 'system_pred']
for col in required_cols:
    if col not in df.columns:
        print(f"  [Warning] '{col}' column missing after renaming. Creating empty placeholder.")
        df[col] = np.nan

# Convert all processing columns to numeric (coerce text/blank cells to NaN)
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nRaw data preview after robust renaming:")
print(df[['timestamp', 'raw_body_temp', 'raw_respi', 'env_temp', 'system_pred', 'raw_state']].head())


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — DATA VALIDATION
#  Check sensor plausibility BEFORE any correction is applied.
# ─────────────────────────────────────────────────────────────────────────────
# Fix blank states by defaulting NaN/blanks to 3 (NORMAL/OFF)
df['raw_state'] = df['raw_state'].fillna(3).astype(int)

print("\n" + "=" * 60)
print("STEP 2: Data Validation")
print("=" * 60)

validation_flags = pd.DataFrame(index=df.index)

# Flag 1: Body temp — physiological range of raw sensor reading
validation_flags['body_out_of_range'] = ~df['raw_body_temp'].between(20.0, 40.0)

# Flag 2: Body temp — sensor floating (reads ambient air, not skin)
validation_flags['body_floating'] = (df['raw_body_temp'] - df['env_temp']) < MIN_CONTACT_DELTA

# Flag 3: Respiratory rate — outside sensor detection range
validation_flags['respi_out_of_range'] = ~df['raw_respi'].between(RESP_MIN, RESP_MAX)

# Flag 4: Environmental temp — impossible reading
validation_flags['env_out_of_range'] = ~df['env_temp'].between(15.0, 45.0)

# Flag 5: State value — must be 0, 1, 2, or 3
validation_flags['state_invalid'] = ~df['raw_state'].isin([0, 1, 2, 3])

# Flag 6: Missing values
validation_flags['has_nan'] = df[['raw_body_temp', 'raw_respi', 'env_temp']].isnull().any(axis=1)

print("Validation summary (True = problem found):")
print(validation_flags.sum().to_string())
print(f"\nRows with ANY flag : {validation_flags.any(axis=1).sum()} / {len(df)}")
print(f"Rows fully clean   : {(~validation_flags.any(axis=1)).sum()} / {len(df)}")

df['validation_flags'] = validation_flags.apply(
    lambda row: [col for col, val in row.items() if val], axis=1
).apply(lambda x: ', '.join(x) if x else 'CLEAN')

print("\nPer-row validation:")
print(df[['raw_body_temp', 'raw_respi', 'env_temp', 'validation_flags']].to_string())


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — DATA PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Data Processing")
print("=" * 60)

# ── 3A: Body Temperature Correction ──────────────────────────────────────────
def correct_body_temp(raw, env):
    delta = raw - env
    if pd.isna(raw) or pd.isna(env):
        return np.nan
    if delta < MIN_CONTACT_DELTA:
        return np.nan   # sensor floating — reject this reading
    t_corrected = raw + CORE_OFFSET + CONTACT_K * delta
    if not (BODY_TEMP_MIN <= t_corrected <= BODY_TEMP_MAX):
        return np.nan   # outside physiological bounds — reject
    return round(t_corrected, 2)

df['corrected_body_temp'] = df.apply(
    lambda r: correct_body_temp(r['raw_body_temp'], r['env_temp']), axis=1
)

# Forward-fill and backward-fill rejected readings
df['corrected_body_temp'] = df['corrected_body_temp'].ffill().bfill()

print(f"Body temp — raw   mean ± std: {df['raw_body_temp'].mean():.2f} ± {df['raw_body_temp'].std():.2f} °C")
print(f"Body temp — fixed mean ± std: {df['corrected_body_temp'].mean():.2f} ± {df['corrected_body_temp'].std():.2f} °C")

# ── 3B: Respiratory Rate Cleaning ────────────────────────────────────────────
def clean_respi(val):
    if pd.isna(val):
        return np.nan
    return float(np.clip(val, RESP_MIN, RESP_MAX))

df['cleaned_respi'] = df['raw_respi'].apply(clean_respi)

# Rolling median over 3 consecutive readings to smooth single-sample spikes
df['cleaned_respi'] = df['cleaned_respi'].rolling(window=3, min_periods=1, center=True).median().round(1)

print(f"Resp rate — raw   mean ± std: {df['raw_respi'].mean():.1f} ± {df['raw_respi'].std():.1f} bpm")
print(f"Resp rate — fixed mean ± std: {df['cleaned_respi'].mean():.1f} ± {df['cleaned_respi'].std():.1f} bpm")

# ── 3C: State Decoding ────────────────────────────────────────────────────────
df['state_label'] = df['raw_state'].map(STATE_LABELS).fillna('UNKNOWN')

# ── 3D: Feature summary ───────────────────────────────────────────────────────
print("\nCleaned data snapshot:")
cols_to_show = ['corrected_body_temp', 'cleaned_respi', 'env_temp', 'state_label']
print(df[cols_to_show].to_string())


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — GROUND TRUTH LABELING (2-of-3 rule matching canine_model.h)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Ground Truth Labeling (2-of-3 rule)")
print("=" * 60)

def label_2of3(row):
    stressed = 0
    if row['corrected_body_temp'] >  BODY_TEMP_THRESHOLD: stressed += 1
    if row['cleaned_respi']      >= RESP_RATE_THRESHOLD:  stressed += 1
    if row['env_temp']            >  ENV_TEMP_THRESHOLD:   stressed += 1
    return 1 if stressed >= 2 else 0

df['ground_truth'] = df.apply(label_2of3, axis=1)

# Derive "is cooling active" from state (states 0,1,2 = cooling ON; state 3 = OFF)
df['cooling_active'] = df['raw_state'].apply(lambda s: 0 if s == 3 else 1)

print("Ground truth label distribution:")
print(df['ground_truth'].value_counts().rename({0: 'NORMAL (0)', 1: 'HEAT_STRESSED (1)'}))

print("\nPer-sensor stress flags:")
print(f"  Body temp  > {BODY_TEMP_THRESHOLD}°C : {(df['corrected_body_temp'] > BODY_TEMP_THRESHOLD).sum()} rows")
print(f"  Resp rate >= {RESP_RATE_THRESHOLD} bpm : {(df['cleaned_respi'] >= RESP_RATE_THRESHOLD).sum()} rows")
print(f"  Env temp   > {ENV_TEMP_THRESHOLD}°C : {(df['env_temp'] > ENV_TEMP_THRESHOLD).sum()} rows")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — PERFORMANCE EVALUATION & METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Performance Evaluation & Metrics")
print("=" * 60)

# If system predictions are still NaN, set to 0 as fallback
df['system_pred'] = df['system_pred'].fillna(0).astype(int)

# Drop any rows where ground_truth or system_pred is missing to ensure clean math
eval_df = df.dropna(subset=['ground_truth', 'system_pred']).copy()

if len(eval_df) > 0:
    y_true = eval_df['ground_truth'].astype(int).values
    y_pred = eval_df['system_pred'].astype(int).values

    # Calculate True Positives, False Positives, False Negatives, True Negatives
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    tn = sum((y_true == 0) & (y_pred == 0))

    # Precision, Recall, F1 formulas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    print(f"Confusion Matrix:")
    print(f"  True Positives (TP) : {tp} | False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn} | True Negatives (TN) : {tn}")
    print("-" * 40)
    print(f"Accuracy  : {accuracy:.2%}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1-Score  : {f1_score_val:.4f}")
else:
    print("Insufficient data to evaluate model performance.")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — STATE TRANSITION ANALYSIS (system behaviour over time)
# ─────────────────────────────────────────────────────────────────────────────
if 'raw_state' in df.columns:
    print("\n" + "=" * 60)
    print("STEP 6: State Transition Analysis")
    print("=" * 60)
    print("State distribution:")
    print(df['state_label'].value_counts().to_string())

    # Detect state transitions
    df['prev_state'] = df['raw_state'].shift(1)
    transitions = df[df['raw_state'] != df['prev_state']].dropna(subset=['prev_state'])
    
    if len(transitions) > 0:
        print(f"\nState transitions detected: {len(transitions)}")
        for _, row in transitions.iterrows():
            prev_val = row['prev_state']
            curr_val = row['raw_state']
            
            prev_name = STATE_LABELS.get(int(prev_val), '?') if pd.notna(prev_val) else 'UNKNOWN'
            curr_name = STATE_LABELS.get(int(curr_val), '?') if pd.notna(curr_val) else 'UNKNOWN'
            
            print(f"  {prev_name} -> {curr_name}")
    else:
        print("No state transitions detected in this slice of data.")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — EXPORT CLEANED DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving cleaned data")
print("=" * 60)

export_cols = ['corrected_body_temp', 'cleaned_respi', 'env_temp', 'ground_truth']
if 'timestamp'      in df.columns: export_cols.insert(0, 'timestamp')
if 'state_label'    in df.columns: export_cols.append('state_label')
if 'system_pred'    in df.columns: export_cols.append('system_pred')
if 'cooling_active' in df.columns: export_cols.append('cooling_active')

df_export = df[export_cols]
df_export.to_csv("cleaned_real_canine_data.csv", index=False)

print(f"Saved: cleaned_real_canine_data.csv  ({len(df_export)} rows x {len(df_export.columns)} cols)")
print("\nFinal cleaned dataset:")
print(df_export.to_string())