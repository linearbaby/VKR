import pandas as pd
import numpy as np
import librosa as ls
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select, insert

from .callback_dataset import CallbackDataset
from .models import *


class DatabaseConnector:
    def __init__(self, music_location, dbapi_uri, echo=False):
        """
        music_location - path to all audio files
        dbapi_uri - uri for connection to mysql db
        of type: "mysql+pymysql://root:password@127.0.0.1:3306/flask"
        echo - log every transaction [True, False]
        """
        super().__init__()
        self.engine = create_engine(dbapi_uri, echo=echo)
        self.session = Session(self.engine)
        self.music_location = Path(music_location)

        Base.metadata.create_all(self.engine)

    def _get_song(self, idx):
        # song_name = self.music_info.loc[idx].paths
        idx += 1  # because in mysql id == 0 is not a valid id
        stmt = select(MusicInfo.path).where(MusicInfo.id == idx)
        song_name = self.session.scalars(stmt).one()
        song, sample_rate = ls.load(self.music_location / song_name)
        return song

    def get_songs_dataset(self):
        return CallbackDataset(
            # return not only song representation, but also idx
            lambda idx: (self._get_song(idx), idx),
            self.session.query(MusicInfo).count(),
        )

    def get_avalilable_songs(self, music_indicies):
        return list(map(self._get_song, music_indicies))

    def create_map(self, name):
        # in this setup it will recreate table
        self.session.query(MusicEmbedding).delete()

    def load_map(self, name):
        pass

    def save_map(self, name):
        pass

    def insert_to_map(self, map_name, song_id, embedding):
        # check if there are multiple values inserted
        try:
            it = iter(song_id)
        except TypeError:  # scalar inserted
            stmt = insert(MusicEmbedding).values(
                fk_id_music_info=int(song_id) + 1, embedding=embedding
            )
            with self.engine.connect() as conn:
                result = conn.execute(stmt)
                conn.commit()
        else:
            song_id = np.array(song_id)
            embedding = np.array(embedding)
            insert = []
            for id, embed in zip(song_id, embedding):
                # id.item() beacause database interprets numpy.int64 type
                # as binary, so need to cast to int
                insert.append(
                    MusicEmbedding(fk_id_music_info=id.item() + 1, embedding=embed)
                )
            self.session.add_all(insert)
            self.session.commit()

    def _get_embedding(self, map_name, song_id):
        # get embedding in form of string [1, 2, 3], then crop parentheses and
        # pass it to numpy.fromstring
        song_id += 1
        stmt = select(MusicEmbedding.embedding).where(
            MusicEmbedding.fk_id_music_info == int(song_id)
        )
        embedding = self.session.scalars(stmt).one()
        return np.fromstring(embedding[1:-1], sep=" ")

    def get_embeddings_dataset(self, map_name):
        return CallbackDataset(
            # return idx alongside embedding, because it is useful
            # for dataset
            lambda idx: (self._get_embedding(map_name, idx), idx),
            self.session.query(MusicEmbedding).count(),
        )
