import numpy as np
import torch
from utils import index, connector
from sys import getsizeof
import gc
from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="index")
def generate_index(cfg: DictConfig):
    # get training data with memory constraints
    cr = connector.SimpleDFConnector(
        cfg.connector.meta_path,
        cfg.connector.music_info_path,
        cfg.connector.music_location,
    )
    cr.load_map(cfg.connector.music_embeddings_name)
    emb_dataset = cr.get_embeddings_dataset(cfg.connector.music_embeddings_name)
    item = emb_dataset[0]  # (np.array, int64)
    memory_for_item = getsizeof(item[0]) + getsizeof(item[1])
    dim = item[0].shape[0]
    cfg.index.dim = dim
    memory_for_item *= 1.1  # количество памяти на элемент с запасом
    possible_train_samples = int(cfg.memory_footprint_training / memory_for_item)
    possible_train_samples = min(possible_train_samples, len(emb_dataset))
    print(f"Training index with {possible_train_samples} samples:")

    emb_loader = torch.utils.data.DataLoader(
        emb_dataset,
        batch_size=cfg.connector.batch_size,
        shuffle=True,
    )
    embeddings = np.empty((0, cfg.index.dim))
    for embedding, _ in islice(emb_loader, possible_train_samples):
        embeddings = np.vstack(embedding.numpy())

    gc.collect()
    index.train_index(embeddings, cfg.index.dim, cfg.index_dir, cfg.index.index_string)

    gc.collect()
    # populate index
    possible_populate_samples = int(
        cfg.memory_footprint_training
        * cfg.memory_footprint_populating
        / memory_for_item
    )
    possible_populate_samples = min(possible_populate_samples, len(emb_dataset))
    emb_populat_loader = torch.utils.data.DataLoader(
        emb_dataset,
        batch_size=possible_populate_samples,
        shuffle=False,
    )

    # change variable types from tensors to numpy arrays
    emb_populat_loader = map(lambda x: (x[0].numpy(), x[1].numpy()), emb_populat_loader)
    index.populate_index(emb_populat_loader, cfg.index_dir, cfg.index.ivf)

    # test
    q = index.FAISS(cfg.index_dir)
    q.execute_query(np.array([[1, 2, 3]]), 1, 1)


if __name__ == "__main__":
    generate_index()
