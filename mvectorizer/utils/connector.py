import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import librosa as ls
from pathlib import Path


class CallbackDataset(Dataset):
    def __init__(self, getitem_callback, len):
        super().__init__()
        self.len = len
        self.getitem_callback = getitem_callback

    def __getitem__(self, idx):
        return self.getitem_callback(idx)

    def __len__(self):
        return self.len


class SimpleDFConnector:
    def __init__(self, data_path, music_location, music_info_df_name="music_info.csv"):
        """
        data_path - путь к общей папке метаинформации
        music_info_path - относительный пути от meta_path до инфо-дф о музыке формата
            csv[song_id|tracks|paths], delim=","
        music_location - путь к папке, в которой располагаются все музыкальные файлы
        """
        super().__init__()
        self.music_info_df_name = music_info_df_name
        self.music_location = music_location
        self.data_path = Path(data_path)

        self.music_info = pd.read_csv(
            self.data_path / music_info_df_name, index_col="song_id"
        )
        self.embedding_maps = {}

    def _get_song(self, idx):
        song_name = self.music_info.loc[idx].paths
        song, sample_rate = ls.load(Path(self.music_location) / song_name)
        return song, idx

    def get_songs_dataset(self):
        return CallbackDataset(self._get_song, len(self.music_info))

    def get_avalilable_songs(self, music_indicies):
        return list(map(self._get_song, music_indicies))

    def create_map(self, name):
        self.embedding_maps[name] = pd.DataFrame(columns=["embeddings"])
        self.embedding_maps[name].index.name = "song_id"
        self.embedding_maps[name] = self.embedding_maps[name].astype(
            {"embeddings": "object"}
        )

    def load_map(self, name):
        self.embedding_maps[name] = pd.read_csv(
            self.data_path / f"{name}.csv", index_col="song_id"
        )

    def save_map(self, name):
        self.embedding_maps[name].to_csv(self.data_path / f"{name}.csv", header=True)

    def insert_to_map(self, map_name, song_id, embedding):
        # check if there are multiple values inserted
        try:
            it = iter(song_id)
        except TypeError:  # scalar inserted
            self.embedding_maps[map_name].at[song_id, "embeddings"] = embedding
        else:
            song_id = np.array(song_id)
            embedding = np.array(embedding).tolist()
            added_df = pd.DataFrame({"embeddings": embedding}, index=song_id)
            added_df.index.name = "song_id"
            self.embedding_maps[map_name] = pd.concat(
                (self.embedding_maps[map_name], added_df)
            )

    def _get_embedding(self, map_name, song_id):
        # get embedding in form of string [1, 2, 3], then crop parentheses and
        # pass it to numpy.fromstring
        return np.fromstring(
            self.embedding_maps[map_name].loc[song_id].embeddings[1:-1], sep=", "
        )

    def get_embeddings_dataset(self, map_name):
        return CallbackDataset(
            # return idx alongside embedding, because it is useful
            # for dataset
            lambda idx: (self._get_embedding(map_name, idx), idx),
            len(self.embedding_maps[map_name]),
        )
