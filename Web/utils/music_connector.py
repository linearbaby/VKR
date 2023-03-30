import pandas as pd
from pathlib import Path
import numpy as np


def get_music_connector(type, config: dict):
    return DiskMusicConnector(**config)


class DiskMusicConnector:
    def __init__(self, music_info_path, music_location):
        self.music_info = pd.read_csv(music_info_path, index_col="song_id")
        self.music_location = Path(music_location)

    def get_music_locations(self, music_ids: list[int]):
        rel_paths = self.music_info.loc[np.array(music_ids)].paths.to_list()
        return rel_paths
