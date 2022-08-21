from fastapi import FastAPI
<<<<<<< HEAD
from application.classes import Response, Request, SingleResponse
import pandas as pd
import pickle
import re


app = FastAPI(description='API для сопоставления товаров с их эталонами')
with open('data/tf_idf_model.pkl', 'rb') as file:
    model = pickle.load(file)
=======
from classes import Response, Request
import pandas as pd
import pickle

app = FastAPI(description='API для сопоставления товаров с их эталонами')
>>>>>>> main


@app.post('/get_results')
async def get_results(message: Request):
    items = [[item.id, item.name, item.props] for item in message.items]
    df = pd.DataFrame(items, columns=['id', 'name', 'props'])
<<<<<<< HEAD
    index = df['id'].values.tolist()
    print(index[0], type(index[0]))

    texts = preprocess_data(df)
    predictions = model.predict(texts)
    predictions = predictions.tolist()

    response = [SingleResponse(id=ind, reference_id=val) for ind, val in zip(index, predictions)]
    return Response(items=response)


def preprocess_data(df):
    df['string_props'] = df['props'].apply(lambda x: ' '.join(x).lower())
    df['text'] = df['name'] + '. ' + df['string_props']
    df['text'] = df['text'].apply(lambda x: x.replace('\t', ' '))
    return df['text']

=======

    with open('data/tf_idf_model.pkl', 'r')
>>>>>>> main
