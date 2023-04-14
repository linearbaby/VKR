import click
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db_connector.src.models import Base, MusicInfo, MusicEmbedding


@click.command()
@click.option(
    "--path-to-samples",
    envvar="PATH_TO_SAMPLES",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--minfo-location",
    required=False,
    envvar="MINFO_LOCATION",
    type=click.Path(exists=True),
)
@click.option(
    "--type-storage",
    required=True,
    type=click.Choice(["DB", "Local"], case_sensitive=False),
    default="DB",
)
def construct_music_info(path_to_samples, minfo_location, type_storage):
    """
    --path-to-samples - path to location of your audio files,
        relative paths to music pieces will be put in dataframe
    --minfo-location - place to store music_info
    --type-storage - type of connector (Mysql DB or pandas DF)
    """
    genres_path = Path(path_to_samples)
    # song_id,tracks,paths
    paths = [
        x.relative_to(path_to_samples) for x in genres_path.glob("**/*") if x.is_file()
    ]
    tracks = [path.stem for path in paths]
    paths = [str(path) for path in paths]

    # sort paths by track names
    sort_idx = [i[0] for i in sorted(enumerate(tracks), key=lambda x: x[1])]
    tracks = [tracks[idx] for idx in sort_idx]
    paths = [paths[idx] for idx in sort_idx]

    # consrtuct dataframe
    if type_storage == "DB":
        insert_to_db(tracks, paths)
    if type_storage == "Local":
        construct_df(tracks, paths, Path(minfo_location) / "music_info.csv")


def construct_df(tracks, paths, minfo_location):
    # consrtuct dataframe
    music_info = pd.DataFrame({"tracks": tracks, "paths": paths})
    music_info.index.name = "song_id"
    music_info.to_csv(minfo_location)


def insert_to_db(tracks, paths):
    cri = "mysql+pymysql://root:password@127.0.0.1:3306/flask"
    engine = create_engine(cri, echo=True)
    with Session(engine) as session:
        Base.metadata.create_all(engine)
        session.query(MusicEmbedding).delete()
        session.query(MusicInfo).delete()

        ids = range(1, len(tracks) + 1)
        music_info = []
        for (
            id,
            name,
            path,
        ) in zip(ids, tracks, paths):
            music_info.append(MusicInfo(id=id, track_name=name, path=path))

        session.add_all(music_info)
        session.commit()


if __name__ == "__main__":
    construct_music_info()
