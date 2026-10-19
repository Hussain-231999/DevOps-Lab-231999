# CI/CD Pipeline and Project Demo Guide

This guide explains how to run the complete Flight Delay Prediction project,
demonstrate the Angular frontend and Flask backend, trigger GitHub Actions, and
verify the updated GitHub Pages website.

## 1. Project architecture

The project has four important parts:

```text
Angular frontend
        |
        | HTTP POST /predict
        v
Flask backend
        |
        v
Trained machine-learning model

GitHub Actions:
  CI -> validates Python and builds Angular
  CD -> publishes the Angular build to GitHub Pages
```

### Components

- `frontend/`: Angular browser application.
- `backend/app.py`: Flask inference API.
- `LAB4/models/flight_delay_models.pkl`: trained model artifact.
- `.github/workflows/ci.yml`: continuous integration workflow.
- `.github/workflows/cd.yml`: continuous deployment workflow.

GitHub Pages hosts the Angular frontend. It does not run Python processes, so
the Flask API must run locally or on a separate Python-capable server. For the
lab demonstration, run Flask locally and open the Angular app locally as well.

## 2. Prerequisites

Install the following:

- Git
- Python 3.11 or a compatible newer Python version
- NVM for Windows (`nvm-windows`)
- A GitHub account with access to the repository

Check the installations:

```powershell
git --version
py --version
nvm version
node --version
npm --version
```

NVM manages Node.js versions. npm is installed automatically with each Node.js
version selected by NVM; npm is not installed separately.

If Node.js is not currently installed through NVM, open PowerShell as
Administrator and run:

```powershell
nvm list available
nvm install 20.19.0
nvm use 20.19.0
```

If that exact version is unavailable, install the latest available Node.js 20
version shown by `nvm list available`, then select it with `nvm use`.

Close and reopen PowerShell after `nvm use`, then verify:

```powershell
node --version
npm --version
```

## 3. Clone and enter the repository

If the repository is not already on the computer:

```powershell
git clone https://github.com/Hussain-231999/DevOps-Lab-231999.git
cd DevOps-Lab-231999
```

If it is already cloned:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo"
git pull origin main
```

The demo guide is at the repository root:

```text
L:\AU\SEM-7\DevOps\LAB\Lab Repo\CI_CD_DEMO_GUIDE.md
```

It is not inside the `LAB4` folder.

Always confirm that the working tree is clean before starting:

```powershell
git status
```

## 4. Run the Flask backend locally

Open PowerShell 1 at the repository root:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo"
py -m pip install -r backend\requirements.txt
py backend\app.py
```

The API starts at:

```text
http://localhost:5000
```

Keep this terminal running.

### Verify the backend

Open this URL in a browser:

```text
http://localhost:5000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

The prediction endpoint is:

```text
POST http://localhost:5000/predict
```

The Angular frontend sends the flight form data to this endpoint. The backend
engineers the required features, loads the trained model, and returns JSON
containing the prediction and delay probability.

## 5. Run the Angular frontend locally

Open PowerShell 2 at the frontend directory:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo\frontend"
npm install
npm start
```

Open the URL shown by Angular, normally:

```text
http://localhost:4200/
```

Enter flight details and click **Predict delay**. The browser sends the data to
the Flask API running in PowerShell 1 and displays the JSON prediction result.

## 6. Run local validation before committing

From the repository root:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo"
python -m compileall -q backend LAB4
```

To validate the backend endpoints:

```powershell
@'
from backend.app import app

client = app.test_client()

health = client.get("/health")
assert health.status_code == 200

prediction = client.post("/predict", json={
    "MONTH": 6,
    "DAY": 15,
    "DAY_OF_WEEK": 1,
    "AIRLINE": "AA",
    "ORIGIN_AIRPORT": "LAX",
    "DESTINATION_AIRPORT": "JFK",
    "SCHEDULED_TIME": 300,
    "DISTANCE": 2475,
    "SCHEDULED_DEPARTURE_HOUR": 8
})

assert prediction.status_code == 200, prediction.get_json()
assert "prediction" in prediction.get_json()
print("Backend validation passed")
'@ | py -
```

The CI workflow performs equivalent checks on an Ubuntu runner and also builds
the Angular production application.

## 7. Demonstrate a code change

For the presentation, make a visible but safe frontend change. For example,
open:

```text
frontend/src/app/app.component.html
```

Find:

```html
<p>Angular frontend communicating with a Flask API and trained model.</p>
```

Change it to:

```html
<p>Updated CI/CD demo: Angular frontend communicating with a Flask API and trained model.</p>
```

This change is useful for the demonstration because:

1. It is visible on the published website.
2. It does not change the machine-learning logic.
3. It is easy to explain.
4. It allows the updated text to prove that CD deployed the new build.

You can also change the page heading or add a small version label, but avoid
changing the API URL during the CI/CD demonstration.

## 8. Commit and push the change

From the repository root:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo"
git status
git add frontend\src\app\app.component.html
git commit -m "Update Angular demo message"
git push origin main
```

The push to `main` triggers CI automatically.

## 9. Explain the CI workflow

Open the repository's **Actions** tab:

<https://github.com/Hussain-231999/DevOps-Lab-231999/actions>

Open the latest `CI` run and explain these steps:

1. **Checkout repository**: downloads the pushed source code.
2. **Set up Python**: installs Python 3.11 on the GitHub-hosted runner.
3. **Install dependencies**: installs Flask, CORS, NumPy, Pandas,
   scikit-learn, XGBoost, and Matplotlib.
4. **Compile Python files**: checks that the backend and machine-learning
   source files have valid Python syntax.
5. **Validate trained model artifact**: loads the pickle file and confirms that
   its experiment data exists.
6. **Set up Node.js**: installs Node.js for the Angular build.
7. **Install Angular dependencies**: installs packages from `frontend/package.json`.
8. **Build Angular frontend**: creates the production files in
   `frontend/dist/flight-delay-frontend/browser`.

CI is successful only if all validation and build steps pass.

## 10. Explain the CD workflow

After CI succeeds, the `Deploy GitHub Pages` workflow starts automatically.
Open the latest deployment workflow and explain these steps:

1. **Checkout repository**: obtains the same `main` branch source.
2. **Set up Node.js**: prepares the Angular build environment.
3. **Install Angular dependencies**: installs the frontend packages.
4. **Build Angular frontend**: creates the production website.
5. **Configure GitHub Pages**: prepares the repository Pages environment.
6. **Upload website artifact**: uploads the Angular `browser` output.
7. **Deploy to GitHub Pages**: publishes the uploaded artifact.

The CD workflow is configured in:

```text
.github/workflows/cd.yml
```

It waits for successful CI using the `workflow_run` trigger. This prevents a
failed validation or failed Angular build from being published.

## 11. Verify the live website

The published website is:

<https://hussain-231999.github.io/DevOps-Lab-231999/>

After the CD workflow is green:

1. Open the website in a new browser tab.
2. Refresh with `Ctrl+F5` if the old version is cached.
3. Confirm that the changed text is visible.
4. Explain that the visible text proves the new Angular bundle was deployed.

The website may take a short time to update because GitHub Pages and browser
caches need to refresh.

## 12. Explain the Flask/GitHub Pages limitation

GitHub Pages is a static hosting service. It can serve Angular's generated
HTML, CSS, and JavaScript files, but it cannot keep a Flask server running.

For this project:

- GitHub Pages hosts the Angular frontend.
- Flask runs locally during the lab demonstration.
- The trained model is loaded by Flask, not by GitHub Pages.
- Angular sends prediction requests to Flask using HTTP.

The local Angular application currently uses:

```typescript
readonly apiUrl = 'http://localhost:5000';
```

If the Flask API is later deployed to a public server, replace this value with
the public API URL and push another commit. The Angular frontend and Flask API
must also allow cross-origin requests, which is why the backend includes CORS
configuration.

## 13. Short presentation script

Use this sequence during the demo:

1. Show the repository structure: Angular frontend, Flask backend, model, and
   workflows.
2. Run Flask and show `http://localhost:5000/health`.
3. Run Angular and make a prediction from the browser.
4. Change the demo message in `app.component.html`.
5. Run `git add`, `git commit`, and `git push`.
6. Open GitHub Actions and show CI running.
7. Explain the Python validation and Angular production build.
8. Show CD starting after CI succeeds.
9. Open GitHub Pages and show the changed message.
10. Explain that Flask runs locally or on a separate server because GitHub Pages
    only hosts the Angular static frontend.

## 14. Troubleshooting

### Angular page is not updated

- Wait for the CD workflow to finish.
- Refresh with `Ctrl+F5`.
- Confirm that the latest CI and CD runs are green.

### Angular prediction returns a network error

- Confirm Flask is running in the other terminal.
- Open `http://localhost:5000/health`.
- Confirm that the frontend API URL is `http://localhost:5000`.

### Flask cannot find the model

Run the backend from the repository root:

```powershell
cd "L:\AU\SEM-7\DevOps\LAB\Lab Repo"
py backend\app.py
```

The model must exist at:

```text
LAB4/models/flight_delay_models.pkl
```

### CI fails before the Angular build

Open the failed CI job and read the first red step. Common causes are:

- missing Python dependency,
- missing model artifact,
- invalid Python syntax,
- incorrect repository path casing.

### CD fails while configuring Pages

In the repository, open **Settings → Pages** and set the source to
**GitHub Actions**. Then rerun the failed workflow or push a new commit.
