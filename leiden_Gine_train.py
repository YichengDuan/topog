import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import networkx as nx
import igraph as ig
import leidenalg
import json

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from datasets.scenes_cluster import ScenesGCNDataset

# ------------------- Hyperparameters -------------------
ROOT = 'datasets'
RESOLUTION = 0.8
BATCH_SIZE = 1
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100
HIDDEN = 64
RESULT_DIR = './results'

# ------------------- Prepare dataset -------------------
dataset = ScenesGCNDataset(root=ROOT, resolution=RESOLUTION)
dataset = dataset.shuffle()
n = len(dataset)
train_ds = dataset[: int(0.8 * n)]
val_ds   = dataset[int(0.8 * n): int(0.9 * n)]
test_ds  = dataset[int(0.9 * n):]
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ------------------- Model Definition -------------------
class GINENet(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout=0.5):
        super().__init__()
        # MLP for GINEConv
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_feats, hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feats, hidden_feats)
        )
        self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=1)

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_feats, hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feats, hidden_feats)
        )
        self.conv2 = GINEConv(self.mlp2, train_eps=True, edge_dim=1)

        # Final classifier
        self.lin = torch.nn.Linear(hidden_feats, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        return F.log_softmax(self.lin(h), dim=1)

# ------------------- Training -------------------
device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
model = GINENet(
    in_feats=dataset.num_node_features,
    hidden_feats=HIDDEN,
    num_classes=int(dataset.data.y.max().item()) + 1
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

val_errors = []
for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            tot += data.num_nodes
    val_err = 1 - correct / tot
    val_errors.append(val_err)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Err: {val_err:.4f}")

# ------------------- Plot & Save -------------------
os.makedirs(RESULT_DIR, exist_ok=True)
plt.figure()
plt.plot(range(1, EPOCHS+1), val_errors, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Validation Error')
plt.title('Validation Error over Epochs')
plt.grid(True)
fig_path = os.path.join(RESULT_DIR, 'val_error.png')
plt.savefig(fig_path)
plt.close()
print(f"Validation error plot saved to {fig_path}")

# ------------------- Test -------------------
model.eval()
correct = tot = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        tot += data.num_nodes
print(f"Test Accuracy: {correct / tot:.4f}")
