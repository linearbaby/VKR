from utils import model
from db_connector import music_connector_factory
from db_connector import create_cri

from fastapi import FastAPI, Header
import os


app = FastAPI()

recommender_model = model.aquire_model(
    user_connector={
        "type": "DB",
        "cri": create_cri(),
        "hidden_size": 512,
        "momentum": 0.9,
    },
    index_path=os.getenv(
        "GRAD_INDEX_PATH",
        "/home/artem/grad/mvectorizer/index/2023-04-10/populated.index",
    ),
)

con = music_connector_factory(
    type="DB",
    dbapi_uri=create_cri(),
    music_location=os.getenv(
        "GRAD_MUSIC_LOCATION", "/home/artem/grad/mvectorizer/data/gtzan/samples"
    ),
)

emb_map_name = "embeddings"
con.load_map(emb_map_name)


def validate(song_id: int):
    return True


@app.get("/")
def root():
    return {"Root of MEM"}


@app.put("/user/{user_id}")
def add_user(user_id: int):
    recommender_model.add_user(user_id)
    return "created"


@app.post("/user/{user_id}")
def update_user_profile(
    user_id: int,
    song_id: int | None = Header(default=None),
    status: bool | None = Header(default=None),
):
    song_emb = con._get_embedding(emb_map_name, song_id)
    recommender_model.update_user(user_id, song_emb, status)
    return "updated"


@app.get("/user/{user_id}")
def query(user_id: int):
    if validate(user_id):
        return {"eval": recommender_model.get_recommendations(user_id)}
    else:
        return "error"


""" 
query with 
curl --header "song-id: 1" localhost:8000
"""
