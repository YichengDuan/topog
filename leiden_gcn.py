import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import igraph as ig
import leidenalg
import numpy as np

## template for our GCN model and training process


# -----------------------------
# 1. Create a synthetic graph
# -----------------------------
# Here we create an Erdős-Rényi graph with 100 nodes.
G = nx.erdos_renyi_graph(n=100, p=0.05, seed=42)

# ----------------------------------------------
# 2. Run Leiden community detection on the graph
# ----------------------------------------------
# Convert the NetworkX graph to an igraph graph.
ig_G = ig.Graph()
ig_G.add_vertices(list(G.nodes()))
edges = list(G.edges())
ig_G.add_edges(edges)

# Perform Leiden community detection using modularity as quality function.
partition = leidenalg.find_partition(ig_G, leidenalg.ModularityVertexPartition)
communities = partition.membership

# Save community labels into the NetworkX graph
for node, community in zip(G.nodes(), communities):
    G.nodes[node]['community'] = community

# ---------------------------------------------------------
# 3. Create node features using the community information
# ---------------------------------------------------------
# One-hot encode the community membership.
num_communities = max(communities) + 1
features = []
for node in G.nodes():
    community = G.nodes[node]['community']
    one_hot = np.zeros(num_communities)
    one_hot[community] = 1
    features.append(one_hot)
features = np.array(features)
x = torch.tensor(features, dtype=torch.float)

# -----------------------------------------------
# 4. Create semantic labels for each node
# -----------------------------------------------
# For demonstration, we assign random semantic labels from two classes.
# In a real scenario, these labels could be "kitchen", "bedroom", etc.
semantic_classes = ["kitchen", "bedroom"]
num_classes = len(semantic_classes)
y_np = np.random.randint(0, num_classes, size=(len(G.nodes()),))
y = torch.tensor(y_np, dtype=torch.long)

# ---------------------------------------------------
# 5. Build edge index (graph connectivity) for PyG
# ---------------------------------------------------
# Convert the edges from the NetworkX graph into a tensor.
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

# If the graph is undirected, add reverse edges.
edge_index_rev = edge_index[[1, 0], :]
edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

# Create the PyTorch Geometric data object.
data = Data(x=x, edge_index=edge_index, y=y)

# -----------------------------
# 6. Define the GCN model
# -----------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # First convolution layer + ReLU activation.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Dropout for regularization.
        x = F.dropout(x, training=self.training)
        # Second convolution layer to output logits for each class.
        x = self.conv2(x, edge_index)
        return x

# Initialize model, optimizer, and loss function.
model = GCN(input_dim=num_communities, hidden_dim=16, output_dim=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# -----------------------------
# 7. Train the GCN model
# -----------------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# -----------------------------
# 8. Inference with the GCN
# -----------------------------
model.eval()
logits = model(data)
preds = logits.argmax(dim=1)
print("Predicted semantic labels (as class indices):", preds)
# To map indices to semantic names:
predicted_semantics = [semantic_classes[i] for i in preds.tolist()]
print("Predicted semantic labels:", predicted_semantics)