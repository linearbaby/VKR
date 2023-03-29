from utils import model
from utils.user import get_user_connector
from db_connector import SimpleDFConnector

from fastapi import FastAPI, Header
import numpy as np


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
con.load_map(emb_map_name)
user_connector = get_user_connector(type=None)


def validate(song_id: int):
    return True


@app.get("/")
def root():
    return {"Root of MEM"}


@app.put("/user/{user_id}")
def add_user(user_id: int):
    user_connector.add_user(user_id)
    return "created"


@app.post("/user/{user_id}")
def update_user_profile(
    user_id: int,
    song_id: int | None = Header(default=None),
    status: bool | None = Header(default=None),
):
    song_emb = con._get_embedding(emb_map_name, song_id)
    user_connector.update_user(user_id, song_emb, status)
    return "updated"


@app.get("/user/{user_id}")
def query(user_id: int):
    if validate(user_id):
        user_embedding = user_connector.get_user(user_id)
        return {
            "eval": index.search(
                user_embedding[
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
