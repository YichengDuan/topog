import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
from torch_geometric.loader import DataLoader
from datasets.scenes_cluster import ScenesGCNDataset
from leiden_Gcn_train import GCNNet
from leiden_Gine_train import GINENet
from leiden_Sage_train import SageNet
import tqdm
import json

# ------------------- Hyperparameters -------------------
ROOT = "datasets"
RESOLUTION = 0.8
BATCH_SIZE = 1
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100
HIDDEN = 64
DROPOUT = 0.5
RESULT_DIR = "./results"
SEED = 12345

# ------------------- Seed Control -------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def ablation_train_test(config: dict):
    """
    Train and test the model with the given config.
    Args:
        config: dict, config for the dataset
    """

    # ------------------- Prepare dataset -------------------
    torch.manual_seed(SEED)
    dataset = ScenesGCNDataset(root=ROOT, resolution=RESOLUTION, config=config)
    dataset = dataset.shuffle()
    n = len(dataset)
    train_ds = dataset[: int(0.8 * n)]
    val_ds = dataset[int(0.8 * n) : int(0.9 * n)]
    test_ds = dataset[int(0.9 * n) :]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ------------------- Training -------------------
    gcn_model = GCNNet(
        in_feats=dataset.num_node_features,
        hidden_feats=HIDDEN,
        num_classes=int(dataset.data.y.max().item()) + 1,
        dropout=DROPOUT,
    ).to(DEVICE)
    sage_model = SageNet(
        in_feats=dataset.num_node_features,
        hidden_feats=HIDDEN,
        num_classes=int(dataset.data.y.max().item()) + 1,
        dropout=DROPOUT,
    ).to(DEVICE)
    gine_model = GINENet(
        in_feats=dataset.num_node_features,
        hidden_feats=HIDDEN,
        num_classes=int(dataset.data.y.max().item()) + 1,
        dropout=DROPOUT,
    ).to(DEVICE)

    current_config_dict = {
        "config": config,
        "gcn_train_loss": [],
        "gcn_val_acc": [],
        "gcn_test_acc": 0.0,
        "sage_train_loss": [],
        "sage_val_acc": [],
        "sage_test_acc": 0.0,
        "gine_train_loss": [],
        "gine_val_acc": [],
        "gine_test_acc": 0.0,
    }

    for model, model_name in zip(
        [gcn_model, sage_model, gine_model], ["GCN", "SAGE", "GINE"]
    ):
        print(f"Training {model_name} model with config: {config}")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        for epoch in tqdm.tqdm(range(1, EPOCHS + 1)):
            # Training
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            current_config_dict[f"{model_name.lower()}_train_loss"].append(train_loss)
            # Validation
            model.eval()
            correct = tot = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(DEVICE)
                    out = model(data.x, data.edge_index, data.edge_attr)
                    pred = out.argmax(dim=1)
                    correct += int((pred == data.y).sum())
                    tot += data.num_nodes
            val_err = 1 - correct / tot
            current_config_dict[f"{model_name.lower()}_val_acc"].append(1 - val_err)
        # ------------------- Test -------------------
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                out = model(data.x, data.edge_index, data.edge_attr)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                tot += data.num_nodes
        test_acc = correct / tot
        current_config_dict[f"{model_name.lower()}_test_acc"] = test_acc
        print(f"{model_name} Test Accuracy: {test_acc:.4f}, done.")

    json_path = os.path.join(
        RESULT_DIR, f"ablation_config_{config['position']}_{config['community']}_{config['objects']}_{config['edge_weight']}.json"
    )
    with open(json_path, "w") as f:
        json.dump(current_config_dict, f, indent=4)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":

    configs = [
        {'position':1, 'community':1, 'objects':1, 'edge_weight':1},
        {'position':0, 'community':1, 'objects':1, 'edge_weight':1},  # −position
        {'position':1, 'community':0, 'objects':1, 'edge_weight':1},  # −community
        {'position':1, 'community':1, 'objects':0, 'edge_weight':1},  # −objects
        {'position':1, 'community':1, 'objects':1, 'edge_weight':0},  # −edge_weight
        {'position':0, 'community':0, 'objects':0, 'edge_weight':0},  # full ablation
    ]

    for config in configs:
        ablation_train_test(config)
        print(f"Config {config} done.")