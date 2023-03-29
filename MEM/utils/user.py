import numpy as np


def get_user_connector(type, hidden_size=512):
    return SimpleUser(hidden_size)


class SimpleUser:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.users = dict()

    def add_user(self, id):
        self.users[id] = {"amount": 0, "embedding": np.zeros(self.hidden_size)}

    def update_user(self, id, emb, status: bool):
        embedding = self.users[id]["amount"] * self.users[id]["embedding"]

        # liked or disliked song
        mul = 1
        if not status:
            mul = -1

        embedding += mul * emb / np.linalg.norm(emb)
        self.users[id]["amount"] += 1
        self.users[id]["embedding"] = embedding

    def get_user(self, id):
        return self.users[id]["embedding"]
