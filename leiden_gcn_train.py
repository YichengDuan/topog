import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from datasets.scenes_dataset import ScenesGCNDataset

# 加载数据集
dataset = ScenesGCNDataset(root='datasets', resolution=0.8)
# 打乱并分割
dataset = dataset.shuffle()
n = len(dataset)
train_ds = dataset[: int(0.8 * n)]
val_ds   = dataset[int(0.8 * n): int(0.9 * n)]
test_ds  = dataset[int(0.9 * n):]

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1)
test_loader  = DataLoader(test_ds, batch_size=1)

# 定义 GCN
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 实例化模型和优化器
in_feats    = dataset.num_node_features
num_classes = int(dataset.data.y.max().item()) + 1
model = GCN(in_feats, hidden_feats=64, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练与评估函数
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def test(loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        mask = batch.test_mask
        correct += int((pred[mask] == batch.y[mask]).sum())
        tot += int(mask.sum())
    return correct / tot if tot > 0 else 0

# 运行训练循环
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        val_acc = test(val_loader)
        print(f'Epoch {epoch:03d}  Loss {loss:.4f}  ValAcc {val_acc:.4f}')
test_acc = test(test_loader)
print(f'Final Test Acc: {test_acc:.4f}')