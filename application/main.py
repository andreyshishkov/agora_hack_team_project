from fastapi import FastAPI
from classes import Response, Request
import pandas as pd
import pickle

app = FastAPI(description='API для сопоставления товаров с их эталонами')


@app.post('/get_results')
async def get_results(message: Request):
    items = [[item.id, item.name, item.props] for item in message.items]
    df = pd.DataFrame(items, columns=['id', 'name', 'props'])

    with open('data/tf_idf_model.pkl', 'r')