import faiss
import numpy as np

from collections import defaultdict

from .user import get_user_connector


class PredictableTemperature:
    def __init__(self, init_temp=5, flatness_factor=4, bias=2, end_temp=0.2):
        self.x = 0
        self.init_temp = init_temp
        self.end_temp = end_temp
        self.flatness_factor = flatness_factor
        self.steps_before_plateu = flatness_factor * 6
        self.bias = bias

    def calc_temp_tanh(self):
        return (1 - np.tanh(self.x / self.flatness_factor - self.bias)) * (
            self.init_temp - self.end_temp
        ) / 2 + self.end_temp

    def calc_temp_hyperbol(self):
        pass
        # return decrease_coeff / x + end_temp

    # def get_update_temp(self, num_recomendations):
    #     if self.x > self.steps_before_plateu:
    #         return self.end_temp

    #     return self.calc_temp_tanh()

    def get_temp(self):
        if self.x > self.steps_before_plateu:
            return self.end_temp
        return self.calc_temp_tanh()

    def update_temp(self, num):
        self.x += num


class StochasticTemperature:
    ...


class RecommenderEngine:
    def __init__(
        self,
        index_path,
        user_connector,  # dict of parameters to get_user_connector
        index_nprobe=4,
        num_nearest=200,
        default_temperature=5,
        distance_multiplier=15,
        temperature_descend=0.99,
        temperature_type="predictable",  # stochastic / predictable
    ):
        """
        if "stochastic", temperature sampling from normal distribution
        will be encorporated, with \mu linearly decreasing to 0.9 (temperature of
        eperienced user). Else temperature will gradually decrease from default
        user temperature to 0.9 with descend coefficent equal to 0.99
        """
        self.temperature_user = None
        if temperature_type == "predictable":
            self.temperature_user = defaultdict(PredictableTemperature)
        if temperature_type == "stochastic":
            self.temperature_user = defaultdict(StochasticTemperature)

        self.index = faiss.read_index(str(index_path))
        self.index.nprobe = index_nprobe
        self.num_nearest = num_nearest
        self.default_temperature = default_temperature
        self.temperature_descend = temperature_descend

        # is needed because logits from recommendation engine
        # are located in range of [-1; 1] so, exp function doesnt work well
        # and convenient temperature scale is not working
        self.distance_multiplier = distance_multiplier

        self.user_connector = get_user_connector(**user_connector)

    def get_recommendations(self, user_id, temperature=None, num_recomendations=5):
        if not temperature:
            temperature = self.temperature_user[user_id].get_temp()

        user_embedding = self.user_connector.get_user(user_id)
        p_values, recomendations = self.index.search(
            user_embedding[np.newaxis,],
            self.num_nearest,
        )

        # softmax with temperature
        p_values = np.exp(p_values * self.distance_multiplier / temperature)
        p_values = p_values / p_values.sum()

        recomendations = np.random.choice(
            a=np.squeeze(recomendations),
            size=(num_recomendations),
            replace=False,
            p=np.squeeze(p_values),
        )

        return recomendations.tolist()

    def update_user(self, user_id, song_emb, status):
        self.user_connector.update_user(user_id, song_emb, status)
        self.temperature_user[user_id].update_temp(1)

    def add_user(self, user_id):
        self.user_connector.add_user(user_id)


def aquire_model(**model_params):
    model = RecommenderEngine(**model_params)
    return model
