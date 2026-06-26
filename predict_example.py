"""
predict_example.py
-------------------
Example of how the trained model would be used on the IoT device (or a
gateway/Raspberry Pi receiving sensor readings) to decide whether to turn
the active cooling system ON or OFF.

Usage
-----
python predict_example.py --model cooling_svm_model.joblib \
    --body_temp 39.8 --resp_rate 35 --ambient_temp 32
"""

import argparse

import joblib
import pandas as pd

from svm_model import FEATURE_COLUMNS


def predict_cooling_action(model, body_temp: float, resp_rate: float, ambient_temp: float):
    sample = pd.DataFrame(
        [[body_temp, resp_rate, ambient_temp]], columns=FEATURE_COLUMNS
    )
    prediction = model.predict(sample)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample)[0][1]  # probability of class "ON"

    action = "TURN COOLING ON" if prediction == 1 else "KEEP COOLING OFF"
    return prediction, proba, action


def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained cooling SVM")
    parser.add_argument("--model", type=str, default="cooling_svm_model.joblib")
    parser.add_argument("--body_temp", type=float, required=True)
    parser.add_argument("--resp_rate", type=float, required=True)
    parser.add_argument("--ambient_temp", type=float, required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    prediction, proba, action = predict_cooling_action(
        model, args.body_temp, args.resp_rate, args.ambient_temp
    )

    print(f"Body Temp     : {args.body_temp} C")
    print(f"Resp Rate     : {args.resp_rate} cycles/min")
    print(f"Ambient Temp  : {args.ambient_temp} C")
    print(f"Prediction    : {prediction} ({'ON' if prediction == 1 else 'OFF'})")
    if proba is not None:
        print(f"Confidence(ON): {proba:.3f}")
    print(f"--> {action}")


if __name__ == "__main__":
    main()
