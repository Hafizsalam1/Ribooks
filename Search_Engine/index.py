import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from fastapi import FastAPI

app = FastAPI()
data = pd.read_csv("data/goodreads_cleaning1.csv")

@app.get("/books/{name}")
def get_book(name: str):
    model = joblib.load("data/tfidf.joblib")

    query_vec = model.transform([name.lower()])
    tfidf = model.transform(data["bookTitle"])
    similarity = cosine_similarity(query_vec, tfidf).flatten()

    indices = np.argsort(-similarity)
    result = data.iloc[indices]['bookTitle'].head(5)

    return {"book":result.to_list()}
