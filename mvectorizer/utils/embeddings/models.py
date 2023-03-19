import torch


class EmbedMusic(torch.nn.Module):
    """
    Embedding module for projecting Jukebox
    High level features to lower dimension space
    """

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_dim, embed_dim),
                torch.nn.Linear(embed_dim, embed_dim),
            ]
        )

    def forward(self, x):
        for emb in self.embedding_layers:
            x = emb(x)
        return x


class ShallowClassifier(torch.nn.Module):
    """
    Downstream task classifier usable to generate
    good music representations
    """

    def __init__(self, input_dim, embed_dim, output_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embedder = EmbedMusic(input_dim, embed_dim)
        self.classifier = torch.nn.Linear(embed_dim, output_dim)

    def embed(self, x):
        return self.embedder(x)

    def forward(self, x):
        y = self.embedder(x)
        y = self.classifier(y)
        return y
