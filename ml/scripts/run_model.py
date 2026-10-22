import pickle
from pathlib import Path

import pandas as pd


def ask_integer(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a whole number.")


model_path = Path(__file__).resolve().parents[1] / "models" / "flight_delay_models.pkl"
with open(model_path, "rb") as file:
    artifact = pickle.load(file)

experiments = artifact["experiments"]
experiment_name = next(iter(experiments))
experiment = experiments[experiment_name]
model_name = next(iter(experiment["models"]))
model_info = experiment["models"][model_name]

print("Enter flight details:")
new_flight_data = pd.DataFrame(
    [
        {
            "MONTH": ask_integer("Month (1-12): "),
            "DAY": ask_integer("Day (1-31): "),
            "DAY_OF_WEEK": ask_integer("Day of week (1-7): "),
            "AIRLINE": input("Airline code (for example, AA): ").strip().upper(),
            "ORIGIN_AIRPORT": input("Origin airport code (for example, LAX): ").strip().upper(),
            "DESTINATION_AIRPORT": input(
                "Destination airport code (for example, JFK): "
            ).strip().upper(),
            "SCHEDULED_TIME": ask_integer("Scheduled flight time in minutes: "),
            "DISTANCE": ask_integer("Distance in miles: "),
            "SCHEDULED_DEPARTURE_HOUR": ask_integer("Scheduled departure hour (0-23): "),
        }
    ]
)

new_flight_data["IS_PEAK_HOURS"] = new_flight_data[
    "SCHEDULED_DEPARTURE_HOUR"
].isin([6, 7, 8, 16, 17, 18, 19, 20]).astype(int)
new_flight_data["IS_RED_EYE"] = new_flight_data[
    "SCHEDULED_DEPARTURE_HOUR"
].isin([23, 0, 1, 2, 3, 4]).astype(int)
new_flight_data["SCHEDULED_SPEED"] = new_flight_data["DISTANCE"] / new_flight_data[
    "SCHEDULED_TIME"
].clip(lower=1)
new_flight_data["ROUTE"] = (
    new_flight_data["ORIGIN_AIRPORT"]
    + "_"
    + new_flight_data["DESTINATION_AIRPORT"]
)
new_flight_data = new_flight_data[experiment["feature_columns"]]
processed_data = experiment["preprocessor"].transform(new_flight_data)
probability = model_info["model"].predict_proba(processed_data)[:, 1][0]
prediction = int(probability >= model_info["threshold"])

print(f"Delay probability: {probability:.4f}")
print(f"Prediction: {'Delayed' if prediction else 'Not delayed'}")
