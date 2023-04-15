FROM base

# change to workspace
COPY MEM /app
WORKDIR /app

# create embeddings and index
# assume that volume with data is mounted
COPY mvectorizer/conf /app/mvectorizer/conf
COPY mvectorizer/utils /app/mvectorizer/utils
COPY mvectorizer/generate_index.py /app/mvectorizer
COPY mvectorizer/generate_embeddings.py /app/mvectorizer

ENTRYPOINT bash init_and_serve.sh 