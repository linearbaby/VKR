from utils.embeddings.models import ShallowClassifier, EmbedMusic
from utils.embeddings.train_eval import train, gen_embeddings
from utils.embeddings.dataset import EmbeddedGTZANDataset
from utils.connector import SimpleDFConnector
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path


@hydra.main(version_base=None, config_path="conf", config_name="embeddings")
def generate(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    # check if embedder not persists on disk
    artifacts_load_path = Path(cfg.emb["artifacts_load_path"])
    if not cfg.emb.train:
        if not artifacts_load_path.exists():
            print(
                """
                artifacts path not found, and train property 
                set to False, so no processing done
                """
            )
            return
    else:
        print(
            """
            attempting to train model
            """
        )
        # time to train model
        model = ShallowClassifier(
            cfg.model["input_dim"], cfg.model["embed_dim"], cfg.model["output_dim"]
        )
        loss_function = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
        raw_embeddings_dataset = EmbeddedGTZANDataset(cfg.emb["emb_dataset_location"])
        train(
            model=model,
            loss_function=loss_function,
            optim=optim,
            epochs=cfg.emb["epochs"],
            batch_size=cfg.emb["batch_size"],
            raw_embeddings_dataset=raw_embeddings_dataset,
            artifacts_path=cfg.model["artifacts_path"],
        )

    print("loading trained model")
    # time to generate embeddings using connector
    embed_model = EmbedMusic(cfg.model["input_dim"], cfg.model["embed_dim"])
    embed_model.load_state_dict(torch.load(artifacts_load_path / "embed.pkl"))
    con = SimpleDFConnector(
        data_path=cfg.connector["meta_path"],
        music_info_df_name=cfg.connector["music_info_df_name"],
        music_location=cfg.connector["music_location"],
    )

    print("generating embeddings")
    raw_embeddings_dataset = EmbeddedGTZANDataset(cfg.emb["emb_dataset_location"])
    gen_embeddings(
        embed_model=embed_model,
        raw_embeddings_dataset=raw_embeddings_dataset,
        connector=con,
        embed_map_name=cfg.connector["emb_map_name"],
        batch_size=cfg.emb["batch_size"],
    )


if __name__ == "__main__":
    generate()
