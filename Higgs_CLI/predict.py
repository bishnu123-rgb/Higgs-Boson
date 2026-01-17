import argparse
import joblib
import pandas as pd
import numpy as np
import sys

def load_model(model_path):
    print("Loading model...")
    return joblib.load(model_path)

def load_input(csv_path):
    print("Loading input data...")
    return pd.read_csv(csv_path)

def preprocess_input(df):
    print("Preprocessing data...")

    drop_cols = ["Label", "Weight"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    # Replacing encoded missing values
    df = df.replace(-999, np.nan)

    # Median imputation
    df = df.fillna(df.median())

    feature_names = joblib.load("feature_names.pkl")

    try:
        df = df[feature_names]
    except KeyError as e:
        print("\n Feature mismatch error")
        print("Your input CSV does not match training features.")
        print("Missing or extra columns detected.")
        sys.exit(1)

    return df

def predict(model, X, threshold):
    print("Running inference...")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description="Higgs Boson Event Classifier (XGBoost)")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--model", default="xgb_model.pkl", help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default=0.5)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")

    args = parser.parse_args()

    model = load_model(args.model)
    df = load_input(args.input)
    X = preprocess_input(df)

    preds, probs = predict(model, X, args.threshold)

    output_df = df.copy()
    output_df["Higgs_Probability"] = probs
    output_df["Prediction"] = preds
    output_df["Prediction_Label"] = output_df["Prediction"].map({1: "Signal", 0: "Background"})

    output_df.to_csv(args.output, index=False)

    print("\nPrediction complete")
    print(f"Saved results to: {args.output}")
    print("\nSample output:")
    print(output_df[["Higgs_Probability", "Prediction_Label"]].head())

if __name__ == "__main__":
    main()
