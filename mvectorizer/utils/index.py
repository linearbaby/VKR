import os
import gc
import faiss
from pathlib import Path
from faiss.contrib.ondisk import merge_ondisk
from enum import Enum


def train_index(data, d, index_dir, index_string):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.index_factory(d, index_string)
    if not index.is_trained:
        index.train(data)
    assert index.is_trained
    faiss.write_index(index, str(index_dir / "trained.index"))


def populate_index(data_iterable, index_dir, ivf):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if not ivf:  # populate in one pass
        data, idx = list(data_iterable)[0]
        index = faiss.read_index(str(index_dir / "trained.index"))
        index.add(data)
        faiss.write_index(index, str(index_dir / "populated.index"))
    else:  # can request on disk merging
        partition = 0
        for partition, data in enumerate(data_iterable):
            gc.collect()
            data, idx = data
            print(f"adding part of data {partition} to index")
            index = faiss.read_index(str(index_dir / "trained.index"))
            index.add_with_ids(data, idx)
            faiss.write_index(index, str(index_dir / f"block_{partition}.index"))

        # construct the output index
        index = faiss.read_index(str(index_dir / "trained.index"))
        block_fnames = [
            index_dir + f"block_{idx}.index" for idx in range(partition + 1)
        ]
        merge_ondisk(index, block_fnames, str(index_dir / "merged_index.ivfdata"))
        faiss.write_index(index, str(index_dir / "populated.index"))


# testing part
class FAISS:
    def __init__(self, index_path):
        index_path = Path(index_path)
        super().__init__()
        self.index = faiss.read_index(str(index_path / "populated.index"))

    def execute_query(self, query, k, probe):
        self.index.nprobe = probe
        return self.index.search(query, k)
