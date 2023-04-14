import pandas as pd
from pathlib import Path
import numpy as np
import randomname
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select

from db_connector.src.models import MusicInfo


def get_music_connector(type, **config):
    if type == "LocalDisk":
        return DiskMusicConnector(**config)
    if type == "DB":
        return DBMusicConnector(**config)
    return None


class DBMusicConnector:
    def __init__(self, cri, music_location):
        self.engine = create_engine(cri, echo=True)
        self.session = Session(self.engine)

        self.music_location = Path(music_location)
        self.names_domains = {"adj": ("music_theory",), "noun": ("cats", "food")}

    def get_music_info(self, music_ids: list[int]):
        """
        returns related information to music like Author etc.
        """
        results = (
            self.session.query(MusicInfo.path).filter(MusicInfo.id.in_(music_ids)).all()
        )
        rel_paths = [row[0] for row in results]
        print(rel_paths)

        authors = [
            randomname.get_name(**self.names_domains) for i in range(len(rel_paths))
        ]
        music_names = [
            randomname.get_name(**self.names_domains) for i in range(len(rel_paths))
        ]
        genres = [rel_path.split("/")[0] for rel_path in rel_paths]

        return [
            {
                "path": rel_path,
                "author": author,
                "music_name": music_name,
                "genre": genre,
                "id": m_id,
            }
            for rel_path, author, music_name, genre, m_id in zip(
                rel_paths, authors, music_names, genres, music_ids
            )
        ]

    def __del__(self):
        self.session.close()


class DiskMusicConnector:
    def __init__(self, music_info_path, music_location):
        self.music_info = pd.read_csv(music_info_path, index_col="song_id")
        self.music_location = Path(music_location)
        self.names_domains = {"adj": ("music_theory",), "noun": ("cats", "food")}

    def get_music_info(self, music_ids: list[int]):
        """
        returns related information to music like Author etc.
        """
        rel_paths = self.music_info.loc[np.array(music_ids)].paths.to_list()
        authors = [
            randomname.get_name(**self.names_domains) for i in range(len(rel_paths))
        ]
        music_names = [
            randomname.get_name(**self.names_domains) for i in range(len(rel_paths))
        ]
        genres = [rel_path.split("/")[0] for rel_path in rel_paths]

        return [
            {
                "path": rel_path,
                "author": author,
                "music_name": music_name,
                "genre": genre,
                "id": m_id,
            }
            for rel_path, author, music_name, genre, m_id in zip(
                rel_paths, authors, music_names, genres, music_ids
            )
        ]
