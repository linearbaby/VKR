from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
import requests
import os

from .models import User
from . import music_connector


songs = Blueprint("songs", __name__)


@songs.route("/update-songs", methods=["POST"])
@login_required
def update_songs():
    song_id = request.headers["song-id"]
    status = request.headers["status"]

    user_id = current_user.id
    response = requests.post(
        f"http://{os.getenv('GRAD_MEM_HOST', '127.0.0.1')}:8000/user/{user_id}",
        headers={"song-id": song_id, "status": status},
    )
    return ""


@songs.route("/renew-recommendations", methods=["GET"])
def renew_recommendations():
    user_id = current_user.id
    response = requests.get(
        f"http://{os.getenv('GRAD_MEM_HOST', '127.0.0.1')}:8000/user/{user_id}",
    )
    music_ids = json.loads(response.content)["eval"]
    music_info = music_connector.get_music_info(music_ids)

    return jsonify(music_info)
