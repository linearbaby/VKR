cd mvectorizer
python generate_embeddings.py emb.emb_dataset_location="/data" connector=database_connector connector.dbapi_uri="mysql+pymysql://root:password@db/flask"
python generate_index.py connector=database_connector connector.dbapi_uri="mysql+pymysql://root:password@db/flask" index_dir="/index"
cd ../
uvicorn --host 0.0.0.0 main:app