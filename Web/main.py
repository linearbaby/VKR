from flask import Blueprint, render_template
from flask_login import login_required, current_user
import requests
import json
from . import music_connector


main = Blueprint("main", __name__)


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
    music_ids = json.loads(response.content)["eval"]
    music_info = music_connector.get_music_info(music_ids)

    return render_template(
        "profile.html",
        name=current_user.name,
        songs_info=music_info,
    )
