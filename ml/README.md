# Flight Delay Prediction

## Angular, Flask, CI/CD, and GitHub Pages

This repository uses GitHub Actions for continuous integration and deployment:

- `CI` installs the Flask and Angular dependencies, compiles the Python source,
  validates the trained model artifact, and builds the Angular production bundle
  on pushes and pull requests.
- `Deploy GitHub Pages` runs after a successful `CI` workflow and publishes the
  Angular production bundle from `frontend/dist/flight-delay-frontend/browser`.

The published project website is available at:

<https://hussain-231999.github.io/DevOps-Lab-231999/>

The Angular frontend calls the Flask inference API at `http://localhost:5000`.
GitHub Pages hosts the frontend, while the Flask backend must run locally or on a
separate Python-capable server. The original Tkinter application is still
available as a local desktop interface.

## Project structure

```text
frontend/
├── src/app/app.component.ts
├── src/app/app.component.html
├── package.json
└── angular.json
backend/
├── app.py
└── requirements.txt
ml/
├── app/app.py
├── data/flights.csv
├── models/flight_delay_models.pkl
├── requirements.txt
└── scripts/
```

## Run the Flask API and Angular frontend

Start the backend in one PowerShell window:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo"
py -m pip install -r backend/requirements.txt
py backend/app.py
```

Start the Angular development server in another PowerShell window:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo\frontend"
npm install
npm start
```

Open <http://localhost:4200/> and use the prediction form. The API health check
is available at <http://localhost:5000/health>.

## Run the training code

1. Open PowerShell in this folder:

   ```powershell
   cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo\ml"
   ```

2. Install the required packages:

   ```powershell
   py -m pip install -r requirements.txt
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