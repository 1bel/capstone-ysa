"""
data_utils.py
--------------
Thresholds, labeling rule, and synthetic dataset generator for the
"IoT Monitoring of Vital Signs and Active Cooling for Canine Heat Stroke
Prevention Using Binary Classification" thesis.

Sensors used (heart rate intentionally excluded per project scope):
    1. Body Temperature   (deg C)
    2. Respiratory Rate   (cycles/minute)
    3. Ambient Temperature(deg C)

Label rule:
    Cooling system turns ON (label = 1) when AT LEAST 2 of the 3 sensors
    have crossed (passed) their normal range into the heat-stressed zone.
    Otherwise the cooling system stays OFF (label = 0).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Threshold table (from Table 2: Master Threshold Classification Table)
# ---------------------------------------------------------------------------
BODY_TEMP_NORMAL_RANGE = (38.3, 39.2)     # deg C
BODY_TEMP_HEAT_CUTOFF = 39.5              # > 39.5 -> heat-stressed

RESP_RATE_NORMAL_RANGE = (10, 30)         # cycles/min
RESP_RATE_HEAT_RANGE = (29, 64)           # heat-stressed band

AMBIENT_TEMP_NORMAL_RANGE = (20, 30)      # deg C
AMBIENT_TEMP_HEAT_CUTOFF = 30             # > 30 -> heat-stressed


def is_body_temp_stressed(value: float) -> bool:
    return value > BODY_TEMP_HEAT_CUTOFF


def is_resp_rate_stressed(value: float) -> bool:
    lo, hi = RESP_RATE_HEAT_RANGE
    return lo <= value <= hi


def is_ambient_temp_stressed(value: float) -> bool:
    return value > AMBIENT_TEMP_HEAT_CUTOFF


def label_sample(body_temp: float, resp_rate: float, ambient_temp: float) -> int:
    """Return 1 (cooling ON) if >=2 of the 3 sensors are heat-stressed, else 0."""
    flags = [
        is_body_temp_stressed(body_temp),
        is_resp_rate_stressed(resp_rate),
        is_ambient_temp_stressed(ambient_temp),
    ]
    return int(sum(flags) >= 2)


def label_dataframe(df: pd.DataFrame) -> pd.Series:
    """Vectorized labeling for a DataFrame with columns:
    body_temp, respiratory_rate, ambient_temp"""
    return df.apply(
        lambda row: label_sample(
            row["body_temp"], row["respiratory_rate"], row["ambient_temp"]
        ),
        axis=1,
    )


def generate_synthetic_dataset(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset for prototyping/training the SVM before
    real IoT sensor logs are available. Replace this with your actual
    logged sensor data (CSV) once collected from the device.

    Sampling ranges are kept wide enough to cover both normal and
    heat-stressed regions for all 3 sensors, with class balancing.
    """
    rng = np.random.default_rng(seed)

    samples = []
    # Oversample then trim/balance so both classes are well represented.
    raw_n = n_samples * 3
    body_temp = rng.uniform(36.5, 41.5, raw_n)
    resp_rate = rng.uniform(5, 70, raw_n)
    ambient_temp = rng.uniform(15, 40, raw_n)

    df = pd.DataFrame(
        {
            "body_temp": body_temp,
            "respiratory_rate": resp_rate,
            "ambient_temp": ambient_temp,
        }
    )
    df["label"] = label_dataframe(df)

    # Balance classes roughly 50/50 for healthier training.
    n_per_class = n_samples // 2
    df_on = df[df["label"] == 1].sample(
        n=min(n_per_class, (df["label"] == 1).sum()), random_state=seed
    )
    df_off = df[df["label"] == 0].sample(
        n=min(n_per_class, (df["label"] == 0).sum()), random_state=seed
    )

    out = pd.concat([df_on, df_off]).sample(frac=1, random_state=seed).reset_index(drop=True)
    out = out.round({"body_temp": 2, "respiratory_rate": 1, "ambient_temp": 2})
    return out


if __name__ == "__main__":
    data = generate_synthetic_dataset()
    print(data["label"].value_counts())
    data.to_csv("synthetic_sensor_data.csv", index=False)
    print("Saved synthetic_sensor_data.csv with", len(data), "rows")
