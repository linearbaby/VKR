from utils import utils
import torch
import hydra
from omegaconf import DictConfig, OmegaConf


def get_model(model_config : DictConfig):
    if model_config.type == "pytorch-Module":
        # assert model path specified
        try:
            model_config.path
        except:
            print("model path not specified!")
            exit()
        
        model = torch.load(model_config.path, map_location='cpu')
        return model
    else:
        exit()
    

def get_embeddings(emb_config: DictConfig):
    pass

@hydra.main(version_base=None, config_path="conf", config_name="embeddings")
def generate(cfg : DictConfig) -> None:
    # model = get_model(cfg.model)
    # get_embeddings(model, cfg.emb)
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    generate()