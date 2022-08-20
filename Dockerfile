# syntax=docker/dockerfile:1
FROM python:3.9
ADD main.py .
ADD ./model_weights/best_model_cnn.h5 .
ADD ./agora_hack_products/agora_hack_products.json
ADD ./agora_hack_products/tokinaizer.json

RUN pip install pandas numpy tqdm tensorflow matplotlib nltk pymorphy2 sklearn
CMD ["python", "./main.py"] 