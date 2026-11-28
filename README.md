# SMS Spam Detection API

A small FastAPI service that classifies English SMS messages as `spam` or
`not_spam`, with a Streamlit interface and MLflow experiment tracking and model
registry. The model combines TF-IDF and logistic regression in one saved
scikit-learn pipeline.

## Start everything with Docker Compose

Start Docker Desktop. Stop any old API container and local API, Streamlit, or
MLflow servers using ports 8000, 8501, or 5000. From this project folder, run:

```powershell
docker compose up --build
```

This starts all three services; no Python environment activation or separate
terminals are needed:

| Service | URL |
| --- | --- |
| Streamlit interface | http://localhost:8501 |
| FastAPI / Swagger | http://localhost:8000/docs |
| MLflow dashboard | http://localhost:5000 |

Streamlit calls the API through the Compose network. It waits for the API's
health check before starting.

On the first startup, MLflow downloads the UCI dataset and trains/registers a
model so the dashboard includes an experiment, metrics, artifacts, and a model
version. This requires internet and may take a minute. A successful setup is
marked in its persistent Docker volume, so later starts do not retrain.
MLflow's database and artifacts survive `docker compose down` and are separate
from the existing Windows `mlflow.db` and `mlruns/`; local history is preserved.
The API serves the `model.joblib` included when the image was built. Model
registration does not automatically replace the API's model.

For later starts without code changes, use `docker compose up`. To run in the
background, use `docker compose up -d`. Check services with `docker compose ps`
and logs with `docker compose logs`. To stop them:

```powershell
docker compose down
```

Rebuild with `docker compose up --build` after changing code or `model.joblib`.
To log another training run in the Docker MLflow instance:

```powershell
docker compose exec mlflow python /app/train.py
```

This registers another version; the API continues serving its built-in model.

## Run locally (PowerShell)

Use Python 3.12. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Training downloads the dataset into `data/` on the first run, removes duplicate
messages, and uses an 80/20 stratified split with random seed 42. It prints test
accuracy, precision, recall, and F1, then saves `model.joblib`. Only the training
set is used to fit the pipeline. The API loads the saved pipeline once at startup.
The included model allows running the API without retraining.

Test results for the included model (4,128 training / 1,032 test messages):

| Metric | Result |
| --- | --- |
| Accuracy | 97.6% |
| Spam precision | 88.1% |
| Spam recall | 93.0% |
| Spam F1 | 90.5% |

Open http://localhost:8000/docs to try both endpoints:

- `GET /`: returns `{"message": "ML API is running"}`.
- `POST /predict`: accepts `{"message": "Are we still meeting at six?"}`
  and returns a prediction such as `{"prediction": "not_spam"}`.

Example PowerShell request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body '{"message": "Are we still meeting at six?"}'
```

Missing, empty, or whitespace-only messages return HTTP 422.

## Streamlit interface

Keep the API running on port 8000. Open a second terminal in the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend.py
```

Open http://localhost:8501, enter a message, and click **Check message**. The page
sends the message to FastAPI and displays **Spam** or **Not spam**. Empty input
shows a warning; if the API is unavailable, the page shows an error.

## MLflow tracking and model registry

Each `train.py` run creates a run in the `sms-spam` experiment and logs:

- Parameters: vectorizer, C, class weight, maximum iterations, random seed,
  and test fraction.
- Test metrics: accuracy, spam precision, spam recall, and spam F1.
- Artifacts: `model.joblib` and the MLflow model with an input example/signature.

The pipeline is registered as **sms-spam-classifier**. MLflow assigns version 1
on its first registration and increments the version on subsequent training runs.
Tracking metadata and the registry are saved locally in `mlflow.db`; model
artifacts are saved in `mlruns/`. These generated files are excluded from Git.
After cloning, run `train.py` to create your own experiment and registered model.

To inspect the runs and registered versions, open a third terminal in the project
folder:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000. Select **Experiments > sms-spam** to view the logged
run, or **Models > sms-spam-classifier** to view its versions. The API continues
to use the local `model.joblib` written by training. Restart the API after
retraining so it loads the updated model.

## Run only the API with Docker

Install and start Docker Desktop, then run from the project folder:

```powershell
docker build -t sms-spam-api .
docker run --rm -p 8000:8000 sms-spam-api
```

Use the same URLs and request above. Stop the local server before running the
container on port 8000. The image includes the saved model and needs no dataset
download or training at startup. For all three services together, use Docker
Compose as described above.

Local verification passed for both endpoints, `/docs`, invalid input, Streamlit
predictions and error handling, MLflow parameters/metrics/artifacts, and registered
model loading and predictions. Both Streamlit and MLflow web pages were checked.
The standalone API image was built and started successfully by the user.
Compose configuration and startup-shell syntax, the frontend's configurable API
connection, and isolated MLflow training/registration/artifact downloads passed
local checks. The full Compose stack still needs verification in Docker Desktop;
Docker was unavailable in the editing environment.

## Dataset

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CC84

Source: https://archive.ics.uci.edu/dataset/228/sms

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Training strips surrounding whitespace, removes duplicate messages, and renames
the `ham` label to `not_spam`. This is a demonstration model trained on historical
English SMS data; its test results do not establish performance on other message
types or languages.
