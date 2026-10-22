# Flight Delay Prediction

This repository contains a machine-learning flight-delay prediction application
with an Angular frontend, Flask inference API, and Tkinter desktop interface.

## Repository structure

```text
.
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── backend/
│   ├── app.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── angular.json
│   └── package.json
├── ml/
│   ├── app/                 # Tkinter desktop application
│   ├── data/                # Local training data
│   ├── docs/                # Research documents
│   ├── models/              # Trained model artifact
│   └── scripts/             # Training and CLI prediction scripts
└── examples/                # Small standalone Python examples
```

## Run the complete application locally

Start the Flask API from the repository root:

```powershell
py -m pip install -r backend\requirements.txt
py backend\app.py
```

In a second terminal, start Angular:

```powershell
cd frontend
npm install
npm start
```

Open the Angular application at:

<http://localhost:4200/DevOps-Lab-231999/>

The Flask health endpoint is:

<http://localhost:5000/health>

## Run the ML scripts

The training dataset is local and is intentionally ignored by Git. If
`ml/data/flights.csv` is available, run:

```powershell
cd ml
py -m pip install -r requirements.txt
py scripts/train_model.py
py scripts/run_model.py
```

The trained artifact is written to `ml/models/flight_delay_models.pkl`.

## CI/CD

On every push or pull request targeting `main`, `CI`:

1. Installs Python dependencies.
2. Compiles the Python sources.
3. Validates the trained model artifact.
4. Installs Angular dependencies.
5. Builds the Angular production bundle.

After successful CI on `main`, `Deploy GitHub Pages` builds and publishes the
Angular bundle to:

<https://hussain-231999.github.io/DevOps-Lab-231999/>

GitHub Pages hosts the Angular frontend. Flask must run locally or on a separate
Python-capable server because GitHub Pages cannot run Python processes.
