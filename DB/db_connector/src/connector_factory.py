from .simple_df_connector import SimpleDFConnector
from .database_connector import DatabaseConnector


def music_connector_factory(type, **kwargs):
    """
    type = ["DB", "DF"]
    kwargs - dict for chosen type of connector
        for DB: > music_location - path to all audio files
                > dbapi_uri - uri for connection to mysql db
                    of type: "mysql+pymysql://root:password@127.0.0.1:3306/flask"

        for DF: > data_path - path to general location of meta information files
                    (in this particular setup meta information - music_info.csv, music_embeddings.csv)
                > music_location - path to all audio files
                > music_info_df_name - name of music info df located in data_path
                    csv[song_id|tracks|paths], delim=","
    """
    if type == "DB":
        return DatabaseConnector(**kwargs)
    if type == "DF":
        return SimpleDFConnector(**kwargs)
    else:
        return None
