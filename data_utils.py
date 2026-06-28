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
BODY_TEMP_HEAT_CUTOFF  = 39.5             # > 39.5 -> heat-stressed

RESP_RATE_NORMAL_RANGE = (10, 30)         # cycles/min
RESP_RATE_HEAT_LOWER   = 29              # >= 29 -> heat-stressed; no upper cap:
                                          # 64 in the table is the observed max,
                                          # NOT a "safe above here" cutoff.

AMBIENT_TEMP_NORMAL_RANGE = (20, 30)      # deg C
AMBIENT_TEMP_HEAT_CUTOFF  = 30            # > 30 -> heat-stressed


def is_body_temp_stressed(value: float) -> bool:
    return value > BODY_TEMP_HEAT_CUTOFF


def is_resp_rate_stressed(value: float) -> bool:
    # Heat-stressed zone starts at >= 29 cycles/min per the threshold table.
    # There is NO upper cap: a dog panting at 70+ cycles/min is still stressed.
    # The table's upper value of 64 is the observed dataset max, not a ceiling.
    return value >= RESP_RATE_HEAT_LOWER


def is_ambient_temp_stressed(value: float) -> bool:
    return value > AMBIENT_TEMP_HEAT_CUTOFF


def validate_sensor_input(body_temp: float, resp_rate: float, ambient_temp: float) -> None:
    """
    Raises ValueError if any reading is physiologically implausible.
    Call this at the system boundary (user input / sensor read) before
    passing values to label_sample() or the trained model.
    """
    if not (30.0 <= body_temp <= 45.0):
        raise ValueError(
            f"Implausible body_temp={body_temp:.2f} C (expected 30-45)"
        )
    if not (0 <= resp_rate <= 150):
        raise ValueError(
            f"Implausible resp_rate={resp_rate:.1f} cycles/min (expected 0-150)"
        )
    if not (-10 <= ambient_temp <= 60):
        raise ValueError(
            f"Implausible ambient_temp={ambient_temp:.2f} C (expected -10 to 60)"
        )


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
    body_temp, respiratory_rate, ambient_temp."""
    body_s    = df["body_temp"]        > BODY_TEMP_HEAT_CUTOFF
    resp_s    = df["respiratory_rate"] >= RESP_RATE_HEAT_LOWER
    ambient_s = df["ambient_temp"]     > AMBIENT_TEMP_HEAT_CUTOFF
    stressed_count = body_s.astype(int) + resp_s.astype(int) + ambient_s.astype(int)
    return (stressed_count >= 2).astype(int)


def generate_synthetic_dataset(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset with realistic physiological correlations:
    high ambient temperature tends to drive up body temperature and respiratory
    rate. Five scenario types cover all quadrants of the 2-of-3 rule so the
    classifier sees every decision region during training.

    Replace this with real IoT sensor logs (CSV) once device data is available.
    """
    rng = np.random.default_rng(seed)

    def make_scenario(n, ambient_range, body_fn, resp_fn):
        ambient = rng.uniform(*ambient_range, n)
        body    = np.clip(body_fn(ambient, rng), 36.5, 41.5)
        resp    = np.clip(resp_fn(ambient, body, rng), 5, 100)
        return pd.DataFrame({
            "body_temp":        body,
            "respiratory_rate": resp,
            "ambient_temp":     ambient,
        })

    counts = {
        "normal":      int(n_samples * 0.30),  # all sensors normal
        "full_stress": int(n_samples * 0.30),  # all sensors stressed
        "amb_resp":    int(n_samples * 0.20),  # ambient+resp stressed, body OK
        "body_resp":   int(n_samples * 0.10),  # body+resp stressed, ambient OK
        "borderline":  int(n_samples * 0.10),  # only 1 sensor stressed -> label 0
    }

    # Scenario 1 — Normal: cool environment, resting dog
    s1 = make_scenario(
        counts["normal"], (20, 30),
        lambda a, rng: rng.uniform(38.3, 39.2, len(a)) + rng.normal(0, 0.1, len(a)),
        lambda a, b, rng: rng.uniform(10, 28, len(a)) + rng.normal(0, 1.5, len(a)),
    )

    # Scenario 2 — Full heat stress: hot environment drives up all three sensors
    s2 = make_scenario(
        counts["full_stress"], (30.5, 40),
        lambda a, rng: 39.6 + (a - 30.5) * 0.12 + rng.normal(0, 0.2, len(a)),
        lambda a, b, rng: 29 + (a - 30.5) * 1.5  + rng.normal(0, 3.0, len(a)),
    )

    # Scenario 3 — Ambient+resp stressed, body still OK (early heat exposure)
    s3 = make_scenario(
        counts["amb_resp"], (30.5, 38),
        lambda a, rng: rng.uniform(38.3, 39.2, len(a)) + rng.normal(0, 0.1, len(a)),
        lambda a, b, rng: rng.uniform(29, 60, len(a)),
    )

    # Scenario 4 — Body+resp stressed, cool ambient (fever or intense exercise indoors)
    s4 = make_scenario(
        counts["body_resp"], (20, 30),
        lambda a, rng: rng.uniform(39.6, 41.0, len(a)),
        lambda a, b, rng: rng.uniform(29, 55, len(a)),
    )

    # Scenario 5 — Borderline: exactly one sensor stressed (label = 0)
    n_bl       = counts["borderline"]
    bl_ambient = rng.uniform(20, 30,   n_bl)
    bl_body    = rng.uniform(38.3, 39.4, n_bl)
    bl_resp    = rng.uniform(10, 28,   n_bl)
    which      = rng.integers(0, 3,    n_bl)
    for i in range(n_bl):
        if   which[i] == 0: bl_body[i]    = rng.uniform(39.6, 41.0)
        elif which[i] == 1: bl_resp[i]    = rng.uniform(29, 60)
        else:               bl_ambient[i] = rng.uniform(30.5, 38)
    s5 = pd.DataFrame({
        "body_temp":        np.clip(bl_body,  36.5, 41.5),
        "respiratory_rate": np.clip(bl_resp,  5, 100),
        "ambient_temp":     bl_ambient,
    })

    df = pd.concat([s1, s2, s3, s4, s5], ignore_index=True)
    df["label"] = label_dataframe(df)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df.round({"body_temp": 2, "respiratory_rate": 1, "ambient_temp": 2})


if __name__ == "__main__":
    data = generate_synthetic_dataset()
    print(data["label"].value_counts())
    data.to_csv("synthetic_sensor_data.csv", index=False)
    print("Saved synthetic_sensor_data.csv with", len(data), "rows")
