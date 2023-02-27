import pandas as pd
import numpy as np
import pydub
import librosa

from enum import Enum
from pathlib import Path

class Locations(Enum):
    LOCAL = 1
    S3 = 2
    UNKNOWN = 3


class Songer:
    def __init__(self, location="local", path=None, music_files_relative=None):
        # encode location
        if location == "local":
            self.location = Locations.LOCAL
        else:
            self.location = Locations.UNKNOWN

        assert path is not None
        self.path = Path(path)
        self.mfiles_path = self.path / music_files_relative
        self.music_info_path = self.path / "music_info.csv"

        self.music_paths_path = self.path / "music_paths.csv"
        self.music_locations = pd.read_csv(self.music_paths_path)

    def save_song(self,):
        pass

    def save_index(self, index_part):
        pass

    def save_mapping(self,):
        pass

    def get_song(self, song_id):
        song_path = self.music_locations.loc[song_id].at["paths"]
        song, sample_rate = librosa.load(self.mfiles_path / song_path)
        return song, sample_rate

    def __getitem__(self, song_id):
        return self.get_song(song_id)

    def get_index(self,):
        pass

    def get_mapping(self,):
        pass
        
    def get_songs_data(self,):
        if self.location == Locations.LOCAL:
            songs_df = pd.read_csv(self.music_info_path)
            return songs_df
        else:
            return None

