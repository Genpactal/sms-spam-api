# SMS Spam Detection API

A small FastAPI service that classifies English SMS messages as `spam` or
`not_spam`. The model combines TF-IDF and logistic regression in one saved
scikit-learn pipeline.

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

## Run with Docker

Install and start Docker Desktop, then run from the project folder:

```powershell
docker build -t sms-spam-api .
docker run --rm -p 8000:8000 sms-spam-api
```

Use the same URLs and request above. Stop the local server before running the
container on port 8000. The image includes the saved model and needs no dataset
download or training at startup.

Local verification passed for both endpoints, `/docs`, and invalid input.
Docker build and container execution have not been verified because Docker was
not installed in the development environment.

## Dataset

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CC84

Source: https://archive.ics.uci.edu/dataset/228/sms

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Training strips surrounding whitespace, removes duplicate messages, and renames
the `ham` label to `not_spam`. This is a demonstration model trained on historical
English SMS data; its test results do not establish performance on other message
types or languages.
