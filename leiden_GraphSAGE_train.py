from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv
import torch, torch.nn.functional as F

# 1. 导入数据集
from datasets.scenes_cluster import ScenesClusterDataset

dataset = ScenesClusterDataset(root="datasets", resolution=0.8)
# 打乱后按 80/10/10 拆分
torch.manual_seed(42)
dataset = dataset.shuffle()
n = len(dataset)
train_ds   = dataset[: int(0.8 * n)]
val_ds     = dataset[int(0.8*n): int(0.9*n)]
test_ds    = dataset[int(0.9*n): ]

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)

# 2. 定义模型
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GraphSAGE(
    in_channels = dataset.num_node_features,
    hidden_channels = 64,
    num_classes = int(dataset.data.y.max().item()) + 1
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 3. 训练与评估
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def eval(loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        tot += batch.num_graphs
    return correct / tot

best_val_acc = 0.0
for epoch in range(1, 101):
    loss = train()
    val_acc = eval(val_loader)
    print(f"Epoch {epoch:03d}  Loss: {loss:.4f}  Val: {val_acc:.4f}")
    # save the model if validation accuracy improves from the previous best
    if epoch == 1 or val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./topo_best_model.pth")


test_acc = eval(test_loader)
print(f"\nTest Accuracy: {test_acc:.4f}")
