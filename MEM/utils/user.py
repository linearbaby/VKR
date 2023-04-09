import numpy as np


def get_user_connector(type, hidden_size=512):
    return SimpleUser(hidden_size)


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
