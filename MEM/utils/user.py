import numpy as np
from db_connector.src.models import Base, UserMEM
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def get_user_connector(type, **kwargs):
    """
    for type == "Memory":
        hidden_size = 512,
        momentum = 0.9

    for type == "DB":
        cri = "mysql+pymysql://root:password@127.0.0.1:3306/flask",
        hidden_size = 512,
        momentum = 0.9
    """
    if type == "Memory":
        return SimpleUser(**kwargs)
    if type == "DB":
        return DBUser(**kwargs)
    return None


class DBUser:
    def __init__(self, cri, hidden_size, momentum=0.9, echo=False):
        self.engine = create_engine(cri, echo=echo)
        self.session = Session(self.engine)
        Base.metadata.create_all(self.engine)

        self.hidden_size = hidden_size
        self.momentum = momentum

    def add_user(self, id):
        rand_user = np.random.randn(self.hidden_size)
        rand_user /= np.linalg.norm(rand_user)
        insert_user = UserMEM(id=id, embedding=rand_user)
        self.session.add(insert_user)

    def update_user(self, id, emb, status: bool):
        user = self.session.get(UserMEM, id)
        embedding = self.momentum * np.fromstring(user.embedding[1:-1], sep=" ")

        # liked or disliked song
        mul = 1
        if not status:
            mul = -1

        embedding += mul * emb / np.linalg.norm(emb) * (1 - self.momentum)
        user.embedding = embedding / np.linalg.norm(embedding)
        self.session.commit()

    def get_user(self, id):
        user = self.session.get(UserMEM, id)
        return np.fromstring(user.embedding[1:-1], sep=" ")

    def __del__(self):
        self.session.close()


class SimpleUser:
    def __init__(self, hidden_size, momentum=0.9):
        self.hidden_size = hidden_size
        self.users = dict()
        self.momentum = momentum

    def add_user(self, id):
        rand_user = np.random.randn(self.hidden_size)
        rand_user /= np.linalg.norm(rand_user)
        self.users[id] = {"amount": 0, "embedding": rand_user}

    def update_user(self, id, emb, status: bool):
        embedding = self.momentum * self.users[id]["embedding"]

        # liked or disliked song
        mul = 1
        if not status:
            mul = -1

        embedding += mul * emb / np.linalg.norm(emb) * (1 - self.momentum)
        self.users[id]["amount"] += 1
        self.users[id]["embedding"] = embedding / np.linalg.norm(embedding)

    def get_user(self, id):
        return self.users[id]["embedding"]
