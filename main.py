from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="SMS Spam Detection API")
model = joblib.load(Path(__file__).with_name("model.joblib"))


class Message(BaseModel):
    model_config = {"str_strip_whitespace": True}
    message: str = Field(min_length=1)


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(body: Message):
    return {"prediction": model.predict([body.message])[0]}
