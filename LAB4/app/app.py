import pickle
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "flight_delay_models.pkl"


def load_model_artifact():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


artifact = load_model_artifact()
preprocessor = artifact["preprocessor"]
model = artifact["models"]["XGBoost"]
threshold = artifact["thresholds"]["XGBoost"]
categories = dict(
    zip(
        artifact["categorical_columns"],
        preprocessor.named_transformers_["cat"].named_steps["onehot"].categories_,
    )
)


def options_for(column):
    if column in categories:
        return [str(value) for value in categories[column]]
    if column == "MONTH":
        return [str(value) for value in range(1, 13)]
    if column == "DAY":
        return [str(value) for value in range(1, 32)]
    if column == "DAY_OF_WEEK":
        return [str(value) for value in range(1, 8)]
    if column == "SCHEDULED_TIME":
        return [str(value) for value in range(30, 721, 5)]
    if column == "DISTANCE":
        return [str(value) for value in range(25, 6001, 25)]
    if column == "SCHEDULED_DEPARTURE_HOUR":
        return [str(value) for value in range(24)]
    raise ValueError(f"No dropdown options configured for {column}")


def predict(entries, result_label):
    try:
        values = {
            "MONTH": int(entries["MONTH"].get()),
            "DAY": int(entries["DAY"].get()),
            "DAY_OF_WEEK": int(entries["DAY_OF_WEEK"].get()),
            "AIRLINE": entries["AIRLINE"].get(),
            "ORIGIN_AIRPORT": entries["ORIGIN_AIRPORT"].get(),
            "DESTINATION_AIRPORT": entries["DESTINATION_AIRPORT"].get(),
            "SCHEDULED_TIME": int(entries["SCHEDULED_TIME"].get()),
            "DISTANCE": int(entries["DISTANCE"].get()),
            "SCHEDULED_DEPARTURE_HOUR": int(
                entries["SCHEDULED_DEPARTURE_HOUR"].get()
            ),
        }
        processed_data = preprocessor.transform(pd.DataFrame([values]))
        probability = model.predict_proba(processed_data)[:, 1][0]
        prediction = "Delayed" if probability >= threshold else "Not delayed"
        result_label.config(
            text=f"Delay probability: {probability:.2%}\nPrediction: {prediction}",
            foreground="#b91c1c" if prediction == "Delayed" else "#166534",
        )
    except (TypeError, ValueError) as error:
        messagebox.showerror("Invalid flight details", str(error))


def create_app():
    window = tk.Tk()
    window.title("Flight Delay Prediction")
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=20)
    frame.grid()
    ttk.Label(
        frame, text="Flight Delay Prediction", font=("Segoe UI", 16, "bold")
    ).grid(row=0, column=0, columnspan=2, pady=(0, 15))

    fields = [
        ("MONTH", "Month"),
        ("DAY", "Day"),
        ("DAY_OF_WEEK", "Day of week"),
        ("AIRLINE", "Airline"),
        ("ORIGIN_AIRPORT", "Origin airport"),
        ("DESTINATION_AIRPORT", "Destination airport"),
        ("SCHEDULED_TIME", "Scheduled time (minutes)"),
        ("DISTANCE", "Distance (miles)"),
        ("SCHEDULED_DEPARTURE_HOUR", "Departure hour"),
    ]
    entries = {}
    for row, (column, label) in enumerate(fields, start=1):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        entry = ttk.Combobox(
            frame, values=options_for(column), state="readonly", width=28
        )
        entry.current(0)
        entry.grid(row=row, column=1, pady=4)
        entries[column] = entry

    result_label = ttk.Label(frame, text="Choose the flight details and click Predict.", padding=(0, 12))
    result_label.grid(row=len(fields) + 1, column=0, columnspan=2)
    ttk.Button(
        frame,
        text="Predict delay",
        command=lambda: predict(entries, result_label),
    ).grid(row=len(fields) + 2, column=0, columnspan=2, sticky="ew")
    return window


if __name__ == "__main__":
    create_app().mainloop()
