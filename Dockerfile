# syntax=docker/dockerfile:1
FROM python:3.9
ADD main.py .
RUN pip install pandas numpy tqdm tensorflow matplotlib nltk pymorphy2 sklearn
CMD ["python", "./main.py"] 