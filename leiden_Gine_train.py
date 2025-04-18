import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv
from datasets.scenes_cluster import ScenesGCNDataset

# 1. 加载数据集
dataset = ScenesGCNDataset(root='datasets', resolution=0.8)
dataset = dataset.shuffle()
n = len(dataset)
train_ds = dataset[: int(0.8 * n)]
val_ds   = dataset[int(0.8 * n): int(0.9 * n)]
test_ds  = dataset[int(0.9 * n):]
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1)
test_loader  = DataLoader(test_ds, batch_size=1)

# 2. 定义 GINEConv 模型
class GINENet(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout=0.5):
        super().__init__()
        # MLP for edge_attr -> message transform
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feats, hidden_feats)
        )
        self.conv1 = GINEConv(self.edge_mlp, train_eps=True)
        self.conv2 = GINEConv(self.edge_mlp, train_eps=True)
        self.lin    = torch.nn.Linear(hidden_feats, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        return F.log_softmax(self.lin(h), dim=1)

# 3. 实例化及训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINENet(
    in_feats=dataset.num_node_features,
    hidden_feats=64,
    num_classes=int(dataset.data.y.max().item()) + 1
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def test(loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr).argmax(dim=1)
        mask = batch.test_mask
        correct += int((pred[mask] == batch.y[mask]).sum())
        tot += int(mask.sum())
    return correct / tot if tot > 0 else 0

best_val_acc = 0.0
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} Loss {loss:.4f} ValAcc {test(val_loader):.4f}")
    # save the best model, if validation accuracy improves from the previous epoch
        if epoch != 1 and test(val_loader) > best_val_acc:
            best_val_acc = test(val_loader)
            torch.save(model.state_dict(), './topo_model/best_gine_model.pth')
            print(f"Saved model with val acc: {best_val_acc:.4f}")




print(f"Final Test Acc: {test(test_loader):.4f}")