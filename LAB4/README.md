# Flight Delay Prediction

## Project structure

```text
lab4/
├── app/app.py
├── data/flights.csv
├── models/flight_delay_models.pkl
├── scripts/train_model.py
├── scripts/run_model.py
└── README.md
```

## Run the training code

1. Open PowerShell in this folder:

   ```powershell
   cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo\lab4"
   ```

2. Install the required packages:

   ```powershell
   py -m pip install numpy pandas scikit-learn xgboost matplotlib
   ```

3. Make sure `data/flights.csv` is present.

4. Run the training code:

   ```powershell
   py scripts/train_model.py
   ```

5. The trained models and preprocessing pipeline will be saved as:

   ```text
   models/flight_delay_models.pkl
   ```

   The artifact contains both the strict pre-departure experiment and the
   target-leakage comparison, including the fitted preprocessing pipeline,
   model thresholds, and feature metadata.

   The Tkinter UI exposes only the strict pre-departure experiment because
   the target-leakage comparison requires `DEPARTURE_DELAY`, which is not
   available before a flight departs.

## Run the pickle file

1. Make sure `models/flight_delay_models.pkl` is present.

2. The interactive prediction code is available in `scripts/run_model.py`. It loads the pickle file and asks for the flight details.

3. Run the pickle model from PowerShell:

   ```powershell
   py scripts/run_model.py
   ```

4. Enter the requested flight details when prompted. The script will print the delay probability and prediction.

## Run the user interface

1. Run the Tkinter interface from PowerShell:

   ```powershell
   py app/app.py
   ```

2. Select the experiment and model, choose a value from each flight-detail
   dropdown, and click **Predict delay**. The UI derives the engineered
   features used by the updated training script, including route, scheduled
   speed, peak-hour, and red-eye indicators.