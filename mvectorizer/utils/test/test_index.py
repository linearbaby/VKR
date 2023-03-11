import numpy as np
from mvectorizer.utils import index
import pytest
import os
import shutil
from pathlib import Path


@pytest.fixture(
    scope="module",
    params=[
        {
            "nlist": 20,  # number of voronoi cells (centroids in index)
            "m": 4,  # number of subquantizers (number of vector partitions)
            "k": 2,  # k nearest
            "nbits": 8,  # each subvector encodes with nbits, bucketing
            "probe": 1,  # how many nearest centroids to take
            "index_dir": "data/index1/",  # !!!!!!!!!!REQUIRED TRAILING SLASH
            "shape": (10000, 512),
        },
        {
            "nlist": 10,  # number of voronoi cells (centroids in index)
            "m": 4,  # number of subquantizers (number of vector partitions)
            "k": 2,  # k nearest
            "nbits": 8,  # each subvector encodes with nbits, bucketing
            "probe": 1,  # how many nearest centroids to take
            "index_dir": "data/index2/",  # !!!!!!!!!!REQUIRED TRAILING SLASH
            "shape": (10240, 256),
        },
    ],
)
def pipeline_params(request):
    dataset = np.random.randn(*request.param["shape"])
    request.param.pop("shape", None)
    return dataset, request.param


def test_pipeline(pipeline_params):
    dataset, params = pipeline_params
    dim = dataset.shape[1]
    index.train_index(
        dataset, dim, params["nlist"], params["m"], params["nbits"], params["index_dir"]
    )
    assert os.path.exists(params["index_dir"])
    assert os.path.exists(Path(params["index_dir"]) / "trained.index")

    index.populate_index(
        [(dataset, np.arange(dataset.shape[0]))],
        params["index_dir"],
    )
    assert os.path.exists(params["index_dir"])
    assert os.path.exists(Path(params["index_dir"]) / f"block_0.index")
    assert os.path.exists(Path(params["index_dir"]) / "merged_index.ivfdata")
    assert os.path.exists(Path(params["index_dir"]) / "populated.index")

    q = index.FAISS(params["index_dir"])
    res = q.execute_query(dataset[:2], params["k"], params["probe"])

    # clear test directories
    shutil.rmtree(Path(params["index_dir"]).parts[0])
