import torch
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from pathlib import Path


def train(
    model,
    loss_function,
    optim,
    epochs,
    batch_size,
    raw_embeddings_dataset,
    artifacts_path,
):
    """
    Train function, possible place for adding some logging logic,
    and modify train process
    """
    best_model_state_dict = None
    best_accuracy = 0
    train_dataset, test_dataset = random_split(raw_embeddings_dataset, [0.8, 0.2])

    for epoch in tqdm(range(epochs)):
        train_data, val_data = random_split(train_dataset, [0.8, 0.2])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

        # training stage
        loss_agg = torch.tensor([])
        iter = tqdm(train_loader)
        iter.set_description(f"Training, epoch: {epoch}")
        for batch in iter:
            data, _ = batch
            X, y = data
            y_hat = model(X)
            optim.zero_grad()
            loss = loss_function(y_hat, y)
            loss_agg = torch.cat((loss_agg, torch.tensor([loss.item()])))
            loss.backward()
            optim.step()
            iter.set_postfix({"loss": loss_agg.mean()})

        # validation stage
        with torch.no_grad():
            accuracy_agg = torch.tensor([])
            iter = tqdm(val_loader)
            iter.set_description(f"Validating, epoch: {epoch}")
            for batch in iter:
                data, _ = batch
                X, y = data
                y_hat = model(X)
                accuracy_agg = torch.cat(
                    (
                        accuracy_agg,
                        torch.tensor(
                            [(y == torch.argmax(y_hat, dim=1)).sum() / y.shape[0]]
                        ),
                    )
                )
                iter.set_postfix({"accuracy": accuracy_agg.mean()})

            # memory save best model
            if best_accuracy < accuracy_agg.mean():
                best_model_state_dict = model.state_dict()
                best_accuracy = accuracy_agg.mean()

    model.load_state_dict(best_model_state_dict)
    predicted = []
    true = torch.tensor([]).reshape(0, 1)
    with torch.no_grad():
        for test_batch in tqdm(
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        ):
            data, _ = test_batch
            X, y = data
            predicted.append(model(X))
            true = torch.vstack((true, y.unsqueeze(1)))

    predicted = torch.vstack(predicted)
    predicted = torch.argmax(predicted, 1)
    accuracy = torch.sum(predicted == true.squeeze())
    accuracy = accuracy / predicted.shape[0]
    print(accuracy)

    # save best model to disk
    artifacts_path = Path(artifacts_path)
    artifacts_path.mkdir(exist_ok=True, parents=True)
    torch.save(model.embedder.state_dict(), artifacts_path / "embed.pkl")
    torch.save(best_model_state_dict, artifacts_path / "model.pkl")


def gen_embeddings(
    raw_embeddings_dataset,
    embed_model,
    connector,
    embed_map_name,
    batch_size,
):
    em_dl = DataLoader(raw_embeddings_dataset, batch_size, shuffle=False)
    connector.create_map(embed_map_name)
    with torch.no_grad():
        for data, music_ids in tqdm(em_dl):
            em, _ = data
            prop_embeddings = embed_model(em)
            prop_embeddings = prop_embeddings / torch.linalg.norm(
                prop_embeddings, dim=1
            ).unsqueeze(-1)
            connector.insert_to_map(embed_map_name, music_ids, prop_embeddings)

    connector.save_map(embed_map_name)
