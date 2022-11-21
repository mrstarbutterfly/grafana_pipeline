import json

import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open("cars_pipe.pkl", 'rb') as file:
    model = dill.load(file)

# print(model['model'])


class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


class Prediction(BaseModel):
    id: int
    pred: str
    price: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return {
        "name": "Car price prediction model",
        "author": "Ekaterina Polovneva",
        "version": 1,
        "date": "2022-05-31T19:09:57.993322",
        "type": "RandomForestClassifier",
        "accuracy": 0.9768
    }


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'id': form.id,
        'pred': y[0],
        'price': form.price
    }



