import click
from pathlib import Path
import pandas as pd


@click.command()
@click.option(
    "--path-to-samples",
    envvar="PATH_TO_SAMPLES",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--minfo-location",
    required=True,
    envvar="MINFO_LOCATION",
    type=click.Path(exists=True),
)
def construct_music_info(path_to_samples, minfo_location):
    """
    --path-to-samples - path to location of your audio files,
        relative paths to music pieces will be put in dataframe
    --minfo-location - place to store music_info
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
    music_info = pd.DataFrame({"tracks": tracks, "paths": paths})
    music_info.index.name = "song_id"
    music_info.to_csv(Path(minfo_location) / "music_info.csv")


if __name__ == "__main__":
    construct_music_info()
