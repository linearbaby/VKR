from flask import Blueprint, render_template
from flask_login import login_required, current_user
import requests
import json

from .utils.music_connector import get_music_connector

main = Blueprint("main", __name__)


music_connector = get_music_connector(
    "LocalDisk",
    {
        "music_info_path": "/home/artem/grad/mvectorizer/data/gtzan/music_info.csv",
        "music_location": "/home/artem/grad/mvectorizer/data/gtzan/samples",
    },
)


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/profile")
@login_required
def profile():
    user_id = current_user.id
    response = requests.get(
        f"http://127.0.0.1:8000/user/{user_id}",
    )
    music_ids = json.loads(response.content)["eval"][0]
    music_paths = music_connector.get_music_locations(music_ids)
    return render_template(
        "profile.html",
        name=current_user.name,
        songs_info=[
            {"id": m_id, "path": m_path} for m_id, m_path in zip(music_ids, music_paths)
        ],
    )
