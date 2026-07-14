"""
clean_data.py
─────────────────────────────────────────────────────────────────────────────
Real Canine Data Cleaning, Validation, Processing & Model Evaluation
Thesis: IoT Monitoring of Vital Signs and Active Cooling for
        Canine Heat Stroke Prevention Using Binary Classification
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1Fc4PQagPnOMZhwk4XmsJaKuyViKE2N2FE92HhPGMERY/export?format=csv&gid=272419857"

# —— Body temperature correction parameters ────────────────────────────────────
# Set APPLY_BODY_TEMP_CORRECTION = True  if sensor contact was poor (loose on fur)
#                                = False if sensor contact was reliable (use raw value)
APPLY_BODY_TEMP_CORRECTION = False

CORE_OFFSET       = 9.0   # °C fixed core-surface gradient (used only when True)
CONTACT_K         = 0.25  # amplification of contact delta (used only when True)
MIN_CONTACT_DELTA = 0.0   # contact quality floor (used only when True)
BODY_TEMP_MIN     = 37.0  # °C core temperature floor (used when correction is ON)
BODY_TEMP_MAX     = 42.5  # °C core temperature ceiling (used when correction is ON)
SKIN_TEMP_MIN     = 28.0  # °C skin surface floor (used when correction is OFF)
SKIN_TEMP_MAX     = 38.0  # °C skin surface ceiling (used when correction is OFF)

# —— Threshold table (Table 2 — Revised) ──────────────────────────────────────
BODY_TEMP_THRESHOLD = 39.5   # > this -> stressed
RESP_RATE_THRESHOLD = 200.0  # >= this -> stressed (panting zone)
ENV_TEMP_THRESHOLD  = 30.0   # > this -> stressed

# —— Respiratory rate physiological bounds ─────────────────────────────────────
RESP_MIN = 50    # bpm — sensor normal lower bound
RESP_MAX = 400   # bpm — maximum panting rate

# —— State mapping (aligned with veryfinalfinalistcode.ino) ────────────────────
STATE_LABELS = {
    0: "HEAT_STRESSED",
    1: "COOLDOWN_PELTIER",
    2: "COOLDOWN_FAN",
    3: "NORMAL"
}

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD AND DYNAMICALLY PARSE MULTI-SUBJECT DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data from Google Sheets...")
print("=" * 60)

df_raw = pd.read_csv(SHEET_CSV_URL)
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

# Rename columns robustly
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
    'timestamp'     : 'timestamp'
}
df_raw = df_raw.rename(columns=rename_dict)

# Parse multiple dogs dynamically using label track matching
parsed_rows = []
current_dog = "unknown"

for idx, row in df_raw.iterrows():
    val_str = str(row['timestamp']).strip().lower()
    # Check if this row is a dog header label zone instead of a timestamp
    if 'rocky' in val_str:
        current_dog = "rocky"
        continue
    elif 'calli' in val_str:
        current_dog = "calli#2"
        continue
    elif 'koba' in val_str:
        current_dog = "koba#3"
        continue
    
    # Skip empty lines or unparsed meta lines
    if pd.isna(row['timestamp']) or val_str == 'nan':
        continue
        
    row_dict = row.to_dict()
    row_dict['dog_name'] = current_dog
    parsed_rows.append(row_dict)

df = pd.DataFrame(parsed_rows).reset_index(drop=True)
print(f"Successfully processed structured canine data matrix: {df.shape[0]} total profiles.")
print(f"Subjects tracked: {df['dog_name'].unique().tolist()}")

# Coerce targeted processing columns to numeric data types safely
required_cols = ['raw_body_temp', 'raw_respi', 'env_temp', 'raw_state', 'system_pred']
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — DATA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
df['raw_state'] = df['raw_state'].fillna(3).astype(int)

print("\n" + "=" * 60)
print("STEP 2: Data Validation")
print("=" * 60)

validation_flags = pd.DataFrame(index=df.index)
validation_flags['body_out_of_range'] = ~df['raw_body_temp'].between(20.0, 45.0)
validation_flags['body_floating']     = (df['raw_body_temp'] - df['env_temp']) < MIN_CONTACT_DELTA
validation_flags['respi_out_of_range']  = ~df['raw_respi'].between(RESP_MIN, RESP_MAX) & df['raw_respi'].notna()
validation_flags['env_out_of_range']    = ~df['env_temp'].between(15.0, 45.0)
validation_flags['state_invalid']       = ~df['raw_state'].isin([0, 1, 2, 3])
validation_flags['has_nan']             = df[['raw_body_temp', 'env_temp']].isnull().any(axis=1)

df['validation_flags'] = validation_flags.apply(
    lambda row: [col for col, val in row.items() if val], axis=1
).apply(lambda x: ', '.join(x) if x else 'CLEAN')

print(f"Rows fully clean : {(~validation_flags.any(axis=1)).sum()} / {len(df)}")
print(f"Rows with issues : {validation_flags.any(axis=1).sum()} / {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — DATA PROCESSING (Correction & Temporal Filtering)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Data Processing Pipeline")
print("=" * 60)

# 3A: Body Temperature Correction
def correct_body_temp(raw, env):
    if pd.isna(raw) or pd.isna(env):
        return np.nan
    if not APPLY_BODY_TEMP_CORRECTION:
        # Sensor contact is reliable — use raw value, validate skin surface bounds only
        return round(raw, 2) if (SKIN_TEMP_MIN <= raw <= SKIN_TEMP_MAX) else np.nan
    # Thermal resistance correction for loose/poor-contact sensor
    delta = raw - env
    if delta < MIN_CONTACT_DELTA:
        return np.nan
    t_corrected = raw + CORE_OFFSET + CONTACT_K * delta
    if not (BODY_TEMP_MIN <= t_corrected <= BODY_TEMP_MAX):
        return np.nan
    return round(t_corrected, 2)

df['corrected_body_temp'] = df.apply(lambda r: correct_body_temp(r['raw_body_temp'], r['env_temp']), axis=1)
df['corrected_body_temp'] = df.groupby('dog_name', group_keys=False).apply(lambda g: g['corrected_body_temp'].ffill().bfill())

# 3B: Respiratory Rate Outlier Clipping & 3-Sample Rolling Median
df['cleaned_respi'] = df['raw_respi'].apply(lambda v: np.nan if pd.isna(v) else float(np.clip(v, RESP_MIN, RESP_MAX)))
df['cleaned_respi'] = df.groupby('dog_name', group_keys=False).apply(lambda g: g['cleaned_respi'].ffill().bfill())
df['cleaned_respi'] = df.groupby('dog_name', group_keys=False).apply(
    lambda g: g['cleaned_respi'].rolling(window=3, min_periods=1, center=True).median().round(1)
)

# 3C: State Structural Decoding
df['state_label'] = df['raw_state'].map(STATE_LABELS).fillna('UNKNOWN')

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — GROUND TRUTH LABELING (2-of-3 Decision Matrix Rule)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Ground Truth Labeling (2-of-3 Rule Engine)")
print("=" * 60)

def label_2of3(row):
    stressed_indicators = 0
    if row['corrected_body_temp'] >  BODY_TEMP_THRESHOLD: stressed_indicators += 1
    if row['cleaned_respi']       >= RESP_RATE_THRESHOLD:  stressed_indicators += 1
    if row['env_temp']            >  ENV_TEMP_THRESHOLD:   stressed_indicators += 1
    return 1 if stressed_indicators >= 2 else 0

df['ground_truth'] = df.apply(label_2of3, axis=1)
df['cooling_active'] = df['raw_state'].apply(lambda s: 0 if s == 3 else 1)

print(df['ground_truth'].value_counts().rename({0: 'NORMAL (0)', 1: 'HEAT_STRESSED (1)'}))

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — HARDWARE PERFORMANCE PERFORMANCE EVALUATION 
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Embedded Machine Learning Performance Evaluation")
print("=" * 60)

df['system_pred'] = df['system_pred'].fillna(0).astype(int)
eval_df = df.dropna(subset=['ground_truth', 'system_pred']).copy()

if len(eval_df) > 0:
    y_true = eval_df['ground_truth'].values
    y_pred = eval_df['system_pred'].values

    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    tn = sum((y_true == 0) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Confusion Matrix Matrix Output:")
    print(f"  True Positives (TP) : {tp} | False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn} | True Negatives (TN) : {tn}")
    print("-" * 50)
    print(f"System Operational Accuracy  : {accuracy:.2%}")
    print(f"System Engineering Precision : {precision:.2%}")
    print(f"System Evaluation Recall     : {recall:.2%}")
    print(f"Calculated F1-Score Value    : {f1_score_val:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — MULTI-SUBJECT SNAPSHOT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: System Output Data Matrix Export")
print("=" * 60)

export_cols = ['timestamp', 'dog_name', 'corrected_body_temp', 'cleaned_respi', 'env_temp', 'state_label', 'system_pred', 'ground_truth']
df_export = df[export_cols]
df_export.to_csv("cleaned_real_canine_data.csv", index=False)

print(f"Saved: cleaned_real_canine_data.csv ({len(df_export)} structural rows)")
print(df_export.to_string())