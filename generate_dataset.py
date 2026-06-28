"""
generate_dataset.py
--------------------
Generates and saves the synthetic canine sensor dataset to a CSV file.

Usage
-----
python generate_dataset.py
python generate_dataset.py --n 8000 --output my_data.csv --seed 99
"""

import argparse

from data_utils import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic canine sensor dataset")
    parser.add_argument("--n",      type=int, default=4000,                    help="Number of samples")
    parser.add_argument("--output", type=str, default="canine_sensor_data.csv", help="Output CSV path")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_dataset(n_samples=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)

    print(f"Dataset saved  : {args.output}")
    print(f"Shape          : {df.shape}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts().rename({0: "Normal / Cooling OFF (0)", 1: "Heat Stressed / Cooling ON (1)"}))
    print(f"\nFeature statistics:")
    print(df[["body_temp", "respiratory_rate", "ambient_temp"]].describe().round(2))


if __name__ == "__main__":
    main()
