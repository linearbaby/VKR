import faiss
import numpy as np

from collections import defaultdict

from .user import get_user_connector


class PredictableTemperature:
    def __init__(
        self, init_temp=5, decrease_coeff=500, end_temp=0.9, steps_before_plateu=10000
    ):
        self.x = decrease_coeff // init_temp
        self.end_temp = end_temp
        self.decrease_coeff = decrease_coeff
        self.steps_before_plateu = steps_before_plateu

    def get_update_temp(self, num_recomendations):
        if self.x > self.steps_before_plateu:
            return self.end_temp

        self.x += num_recomendations
        return self.decrease_coeff / self.x + self.end_temp


class StochasticTemperature:
    ...


class RecommenderEngine:
    def __init__(
        self,
        index_path,
        index_nprobe=4,
        user_connector_type=None,
        default_temperature=5,
        temperature_descend=0.99,
        enginge_type="predictable",  # stochastic / predictable
    ):
        """
        if "stochastic", temperature sampling from normal distribution
        will be encorporated, with \mu linearly decreasing to 0.9 (temperature of
        eperienced user). Else temperature will gradually decrease from default
        user temperature to 0.9 with descend coefficent equal to 0.99
        """
        self.temperature_user = None
        if enginge_type == "predictable":
            self.temperature_user = defaultdict(PredictableTemperature)
        if enginge_type == "stochastic":
            self.temperature_user = defaultdict(StochasticTemperature)

        self.index = faiss.read_index(str(index_path))
        self.index.nprobe = index_nprobe
        self.num_nearest = 500
        self.default_temperature = default_temperature
        self.temperature_descend = temperature_descend

        self.user_connector = get_user_connector(type=user_connector_type)

    def get_recommendations(self, user_id, temperature=None, num_recomendations=5):
        if not temperature:
            temperature = self.temperature_user[user_id].get_update_temp(
                num_recomendations
            )

        user_embedding = self.user_connector.get_user(user_id)
        recomendations, p_values = self.index.search(
            user_embedding[np.newaxis,],
            self.num_nearest,
        )

        # softmax with temperature
        p_values = np.exp(p_values / temperature)
        p_values = p_values / p_values.sum()

        recomendations = np.random.sample(
            recomendations, size=num_recomendations, replace=False, p=p_values
        )

        return recomendations.to_list()

    def update_user(self, user_id, song_emb, status):
        self.user_connector.update_user(user_id, song_emb, status)

    def add_user(self, user_id):
        self.user_connector.add_user(user_id)


def aquire_model(model_path):
    model = RecommenderEngine(str(model_path))
    return model
