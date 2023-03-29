from utils import model
from fastapi import FastAPI, Header
import numpy as np

from db_connector import SimpleDFConnector


emb_map_name = "embeddings"
nprobe = 4
k = 5

app = FastAPI()
index = model.aquire_model(
    "/home/artem/grad/mvectorizer/index/2023-03-19/populated.index"
)
index.nprobe = nprobe
con = SimpleDFConnector(
    data_path="/home/artem/grad/mvectorizer/data/gtzan",
    music_location="/home/artem/grad/mvectorizer/data/gtzan/samples",
)
con.load_map("embeddings")


def validate(song_id: int):
    return True


@app.get("/")
def root():
    return {"Root of MEM"}


@app.get("/query")
def query(song_id: int | None = Header(default=None)):
    if validate(song_id):
        return {
            "eval": index.search(
                con._get_embedding(emb_map_name, song_id)[
                    np.newaxis,
                ],
                k,
            )[1].tolist()
        }
    else:
        return "error"


""" 
query with 
curl --header "song-id: 1" localhost:8000
"""
