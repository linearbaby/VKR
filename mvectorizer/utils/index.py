import os
import gc
import faiss
from pathlib import Path
from faiss.contrib.ondisk import merge_ondisk


def train_index(data, d, nlist, m, nbits, index_dir):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    index.train(data)
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, index_dir + "trained.index")
    return index


def populate_index(data_iterator, index_dir):
    partition = 0
    for partition, data in enumerate(data_iterator):
        gc.collect()
        data, idx = data
        print(f"adding part of data {partition} to index")
        index = faiss.read_index(index_dir + "trained.index")
        index.add_with_ids(data, idx)
        faiss.write_index(index, index_dir + f"block_{partition}.index")

    # construct the output index
    index = faiss.read_index(index_dir + "trained.index")
    block_fnames = [index_dir + f"block_{idx}.index" for idx in range(partition + 1)]
    merge_ondisk(index, block_fnames, index_dir + "merged_index.ivfdata")
    faiss.write_index(index, index_dir + "populated.index")


# testing part
class FAISS:
    def __init__(self, index_path):
        super().__init__()
        self.index = faiss.read_index(index_path + "populated.index")

    def execute_query(self, query, k, probe):
        self.index.nprobe = probe
        return self.index.search(query, k)
