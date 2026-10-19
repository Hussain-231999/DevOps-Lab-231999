import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(ROOT_DIR / "LAB4" / "models" / "flight_delay_models.pkl"),
    )
)

app = Flask(__name__)
CORS(app)


def load_artifact():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


ARTIFACT = load_artifact()
EXPERIMENT = ARTIFACT["experiments"]["Path B: Strict Pre-Departure"]


def build_features(payload):
    required = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_TIME",
        "DISTANCE",
        "SCHEDULED_DEPARTURE_HOUR",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    hour = int(payload["SCHEDULED_DEPARTURE_HOUR"])
    scheduled_time = max(int(payload["SCHEDULED_TIME"]), 1)
    distance = int(payload["DISTANCE"])
    origin = str(payload["ORIGIN_AIRPORT"]).strip().upper()
    destination = str(payload["DESTINATION_AIRPORT"]).strip().upper()

    if not 1 <= int(payload["MONTH"]) <= 12:
        raise ValueError("MONTH must be between 1 and 12")
    if not 1 <= int(payload["DAY"]) <= 31:
        raise ValueError("DAY must be between 1 and 31")
    if not 1 <= int(payload["DAY_OF_WEEK"]) <= 7:
        raise ValueError("DAY_OF_WEEK must be between 1 and 7")
    if not 0 <= hour <= 23:
        raise ValueError("SCHEDULED_DEPARTURE_HOUR must be between 0 and 23")
    if not origin or not destination:
        raise ValueError("Airport codes cannot be empty")
    if distance <= 0:
        raise ValueError("DISTANCE must be greater than zero")

    values = {
        "MONTH": int(payload["MONTH"]),
        "DAY": int(payload["DAY"]),
        "DAY_OF_WEEK": int(payload["DAY_OF_WEEK"]),
        "AIRLINE": str(payload["AIRLINE"]).strip().upper(),
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": destination,
        "SCHEDULED_TIME": scheduled_time,
        "DISTANCE": distance,
        "SCHEDULED_DEPARTURE_HOUR": hour,
        "IS_PEAK_HOURS": int(hour in [6, 7, 8, 16, 17, 18, 19, 20]),
        "IS_RED_EYE": int(hour in [23, 0, 1, 2, 3, 4]),
        "SCHEDULED_SPEED": distance / scheduled_time,
        "ROUTE": f"{origin}_{destination}",
    }
    return pd.DataFrame(
        [{column: values[column] for column in EXPERIMENT["feature_columns"]}]
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")

        input_data = build_features(payload)
        model_name = next(iter(EXPERIMENT["models"]))
        model_info = EXPERIMENT["models"][model_name]
        processed_data = EXPERIMENT["preprocessor"].transform(input_data)
        probability = float(
            model_info["model"].predict_proba(processed_data)[:, 1][0]
        )
        delayed = probability >= model_info["threshold"]
        return jsonify(
            {
                "prediction": "Delayed" if delayed else "Not delayed",
                "delay_probability": probability,
                "model": model_name,
            }
        )
    except (TypeError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
