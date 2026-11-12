import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "flights.csv"
RESULTS_PATH = ROOT_DIR / "models" / "base_paper_results.json"

# The base paper describes these 19 operational predictors and predicts arrival delay.
BASE_PAPER_FEATURE_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "SCHEDULED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
    "ARRIVAL_TIME",
]
TARGET_COLUMN = "ARRIVAL_DELAY"
PREDEPARTURE_FEATURE_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
]


def load_sample(sample_fraction):
    columns = BASE_PAPER_FEATURE_COLUMNS + [TARGET_COLUMN]
    chunks = []
    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=columns,
        dtype={
            "AIRLINE": str,
            "TAIL_NUMBER": str,
            "ORIGIN_AIRPORT": str,
            "DESTINATION_AIRPORT": str,
        },
        chunksize=500_000,
    ):
        chunks.append(chunk.sample(frac=sample_fraction, random_state=42))

    data = pd.concat(chunks, ignore_index=True).dropna(subset=[TARGET_COLUMN])
    return data


def build_preprocessor(data, feature_columns):
    features = data[feature_columns]
    numeric_columns = features.select_dtypes(
        include="number"
    ).columns.tolist()
    categorical_columns = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=20,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def evaluate_models(data, feature_columns):
    x = data[feature_columns]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    preprocessor = build_preprocessor(data, feature_columns)
    x_train_prepared = preprocessor.fit_transform(x_train)
    x_test_prepared = preprocessor.transform(x_test)

    models = {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            tree_method="hist",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train_prepared, y_train)
        predictions = model.predict(x_test_prepared)
        results[name] = {
            "MAE": round(mean_absolute_error(y_test, predictions), 4),
            "RMSE": round(mean_squared_error(y_test, predictions) ** 0.5, 4),
            "R2": round(r2_score(y_test, predictions), 4),
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the base paper's arrival-delay regression setup."
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.20,
        help="Fraction sampled from each CSV chunk (default: 0.20).",
    )
    args = parser.parse_args()
    if not 0 < args.sample_frac <= 1:
        raise ValueError("--sample-frac must be greater than 0 and at most 1")

    data = load_sample(args.sample_frac)
    results = {
        "base_paper_operational": evaluate_models(data, BASE_PAPER_FEATURE_COLUMNS),
        "strict_pre_departure": evaluate_models(
            data, PREDEPARTURE_FEATURE_COLUMNS
        ),
    }

    artifact = {
        "base_paper": "Flight Delay Prediction Using Machine Learning Algorithm XGBoost",
        "dataset": str(DATA_PATH),
        "sample_fraction_per_chunk": args.sample_frac,
        "base_paper_feature_columns": BASE_PAPER_FEATURE_COLUMNS,
        "strict_pre_departure_feature_columns": PREDEPARTURE_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "split": {"test_size": 0.20, "random_state": 42},
        "metrics": results,
    }
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
