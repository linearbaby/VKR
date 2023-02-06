from utils import model
from fastapi import FastAPI, Header
import numpy as np

app = FastAPI()
eval_model = model.aquire_model()

@app.get("/")
def root():
    return {"Root of MEM"}

@app.get("/query")
def query(query_params: str | None = Header(default=None)):
    query_params = query_params[1:-1]
    query_params = [int(x) for x in query_params.split(",")]
    n_query = np.atleast_2d(query_params)

    return {"eval": eval_model.predict(n_query).tolist()}

''' 
query with 
curl --header "query-params: [1, 2]" localhost:8000
'''
