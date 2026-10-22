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
experiments = artifact["experiments"]
experiments = {
    "Path B: Strict Pre-Departure": experiments["Path B: Strict Pre-Departure"]
}
experiment_names = list(experiments)


def options_for(column):
    if column in {"AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"}:
        categories = set()
        for experiment in experiments.values():
            if column in experiment["categorical_columns"]:
                encoder = (
                    experiment["preprocessor"]
                    .named_transformers_["cat"]
                    .named_steps["onehot"]
                )
                index = experiment["categorical_columns"].index(column)
                categories.update(str(value) for value in encoder.categories_[index])
        return sorted(categories)
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


def predict(entries, experiment_name, model_name, result_label):
    try:
        experiment = experiments[experiment_name]
        model_info = experiment["models"][model_name]
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
        values["IS_PEAK_HOURS"] = int(
            values["SCHEDULED_DEPARTURE_HOUR"] in [6, 7, 8, 16, 17, 18, 19, 20]
        )
        values["IS_RED_EYE"] = int(
            values["SCHEDULED_DEPARTURE_HOUR"] in [23, 0, 1, 2, 3, 4]
        )
        values["SCHEDULED_SPEED"] = values["DISTANCE"] / max(
            values["SCHEDULED_TIME"], 1
        )
        values["ROUTE"] = (
            f"{values['ORIGIN_AIRPORT']}_{values['DESTINATION_AIRPORT']}"
        )
        input_data = pd.DataFrame(
            [{column: values[column] for column in experiment["feature_columns"]}]
        )
        processed_data = experiment["preprocessor"].transform(input_data)
        probability = model_info["model"].predict_proba(processed_data)[:, 1][0]
        prediction = "Delayed" if probability >= model_info["threshold"] else "Not delayed"
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

    ttk.Label(frame, text="Experiment").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=4)
    experiment_entry = ttk.Combobox(
        frame, values=experiment_names, state="readonly", width=28
    )
    experiment_entry.current(0)
    experiment_entry.grid(row=1, column=1, pady=4)

    ttk.Label(frame, text="Model").grid(row=2, column=0, sticky="w", padx=(0, 12), pady=4)
    model_entry = ttk.Combobox(
        frame,
        values=list(experiments[experiment_names[0]]["models"]),
        state="readonly",
        width=28,
    )
    model_entry.current(0)
    model_entry.grid(row=2, column=1, pady=4)

    def update_models(_event):
        model_entry["values"] = list(experiments[experiment_entry.get()]["models"])
        model_entry.current(0)

    experiment_entry.bind("<<ComboboxSelected>>", update_models)

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
    for row, (column, label) in enumerate(fields, start=3):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        entry = ttk.Combobox(
            frame, values=options_for(column), state="readonly", width=28
        )
        entry.current(0)
        entry.grid(row=row, column=1, pady=4)
        entries[column] = entry

    result_label = ttk.Label(frame, text="Choose the flight details and click Predict.", padding=(0, 12))
    result_label.grid(row=len(fields) + 3, column=0, columnspan=2)
    ttk.Button(
        frame,
        text="Predict delay",
        command=lambda: predict(
            entries, experiment_entry.get(), model_entry.get(), result_label
        ),
    ).grid(row=len(fields) + 4, column=0, columnspan=2, sticky="ew")
    return window


if __name__ == "__main__":
    create_app().mainloop()
