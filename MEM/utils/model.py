import faiss


def aquire_model(model_path):
    model = faiss.read_index(str(model_path))
    return model
