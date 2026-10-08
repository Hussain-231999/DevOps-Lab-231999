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

preprocessor = artifact["preprocessor"]
model = artifact["models"]["XGBoost"]
threshold = artifact["thresholds"]["XGBoost"]

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

processed_data = preprocessor.transform(new_flight_data)
probability = model.predict_proba(processed_data)[:, 1][0]
prediction = int(probability >= threshold)

print(f"Delay probability: {probability:.4f}")
print(f"Prediction: {'Delayed' if prediction else 'Not delayed'}")
