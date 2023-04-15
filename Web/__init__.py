from db_connector import create_cri
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

import os

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()

from .utils.music_connector import get_music_connector

music_connector = get_music_connector(
    "DB",
    cri=create_cri(),
    music_location=os.getenv(
        "GRAD_MUSIC_LOCATION", "/home/artem/grad/mvectorizer/data/gtzan/samples"
    ),
)


def create_app():
    app = Flask(
        __name__,
        static_url_path="",
        static_folder=os.getenv(
            "GRAD_MUSIC_LOCATION", "/home/artem/grad/mvectorizer/data/gtzan/samples"
        ),
        template_folder="templates",
    )

    app.config["SECRET_KEY"] = "9OLWxND4o83j4K4iuopO"
    app.config["SQLALCHEMY_DATABASE_URI"] = create_cri()

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from .auth import auth as auth_blueprint

    app.register_blueprint(auth_blueprint)

    from .songs import songs as songs_blueprint

    app.register_blueprint(songs_blueprint)

    # blueprint for non-auth parts of app
    from .main import main as main_blueprint

    app.register_blueprint(main_blueprint)

    with app.app_context():
        db.create_all()

    return app
