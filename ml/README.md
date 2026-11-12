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

## Base paper and research contribution

The selected base paper is
`FlightDelayPredictionUsingMachineLearningAlgorithmXGBoost20192117.pdf`,
*Flight Delay Prediction Using Machine Learning Algorithm XGBoost*.
The paper uses the public 2015 U.S. domestic flight dataset from the Bureau of
Transportation Statistics, applies preprocessing and an 80/20 split, and
compares XGBoost with linear regression techniques.

The repository's `data/flights.csv` is the same underlying 2015 U.S. flight
dataset family. The paper reports a processed dataset with 19 features, while
the current source file contains 31 columns, so the reproduction must document
the selected features, target definition, filtering, and split before reporting
a direct comparison. The current artifact's strict pre-departure XGBoost result
(ROC-AUC 0.7089, F1-score 0.4137) must not be presented as the paper's result
until that protocol is matched.

The intended research contribution is a leakage-aware, pre-departure
evaluation that improves the paper's practical reliability. Path B is the
deployable primary experiment because it excludes `DEPARTURE_DELAY`. Path A is
not a second production model or a requirement of the paper; retain it only as
a separate leakage diagnostic showing how post-departure information inflates
performance.

### Reproduction and comparison workflow

Run the base-paper-aligned regression baseline with:

```powershell
py scripts/reproduce_base_paper.py
```

This uses the base paper's described 19 operational predictors, predicts continuous
`ARRIVAL_DELAY`, uses an 80/20 split, and reports MAE, RMSE, and R2 for Linear
Regression and XGBoost. It also evaluates the same regression task using only
pre-departure predictors. Results are saved to
`models/base_paper_results.json`. The base paper is not sufficiently explicit about
every preprocessing detail or a single reproducible benchmark table, so the
script records its assumptions instead of claiming an exact numerical
reproduction.

The full run currently reports the following on the repository dataset sample:

| Experiment | Model | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| Paper-aligned operational | Linear Regression | 2.7081 | 4.9604 | 0.9840 |
| Paper-aligned operational | XGBoost | 4.9379 | 8.9955 | 0.9473 |
| Strict pre-departure | Linear Regression | 20.8741 | 38.7079 | 0.0248 |
| Strict pre-departure | XGBoost | 20.2981 | 38.1269 | 0.0539 |

The large gap between the operational and strict results is evidence that
post-departure variables, especially `DEPARTURE_DELAY`, make the paper-aligned
task substantially easier. The strict results are the fair comparison for the
proposed leakage-aware contribution; the existing Path B classification
metrics are a separate deployment-oriented task and should not be compared
directly with regression MAE or R2. The current results support a stronger
validity claim, not a claim of higher raw accuracy than the leakage-using
baseline.

The research comparison is therefore:

1. `reproduce_base_paper.py`: base-paper-aligned operational regression baseline.
2. Path B in `train_model.py`: proposed strict pre-departure classifier.
3. Path A: optional leakage diagnostic, excluded from deployment.

The proposed contribution is the explicit separation of deployable
pre-departure information from post-departure leakage, with both protocols and
their limitations reported transparently.

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