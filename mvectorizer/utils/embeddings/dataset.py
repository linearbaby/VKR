from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from librosa import load
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class EmbeddedGTZANDataset(Dataset):
    """
    Jukebox model cannot be fit in small environments, so
    GTZAN dataset already preprocessed with Jukebox encoder is
    presented.
    """

    def __init__(self, data_location):
        super().__init__()
        self.location = Path(data_location)
        np_location = self.location / "features"
        music_info = pd.read_csv(self.location / "music_info.csv")
        self.music_ids = music_info.index.to_numpy()

        labels = [track.split(".")[0] for track in music_info.tracks]
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(labels)
        self.y = torch.tensor(self.y)

        self.X = np.array(
            [
                np.load(np_location / (file_name + ".npy"))
                for file_name in music_info.tracks
            ]
        )
        self.X = torch.tensor(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]), self.music_ids[idx]

    def get_genre_sample(self, genre, idx):
        genre_encoded = list(self.le.classes_).index(genre)
        genre_samples = self.y == genre_encoded
        item_idx = [i for i, n in enumerate(genre_samples) if n][idx]
        return self.X[item_idx]

    def __len__(self):
        return self.X.shape[0]


class EmbeddingsDataset(Dataset):
    """
    Dataset implementation for not preprocessed with JukeBox encoder
    samples of audio
    """

    def __init__(self, music_dataset, JukeBoxEncoder):
        super().__init__()
        self.music_dataset = music_dataset
        self.encoder = JukeBoxEncoder

    def __getitem__(self, idx):
        music, label = self.music_dataset[idx]
        encoded_music = self.encoder(music)
        return encoded_music, label

    def __len__(self):
        return len(self.music_dataset)
