import os, glob
import torch
import networkx as nx
import igraph as ig
import leidenalg

from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx

class ScenesGCNDataset(InMemoryDataset):
    """
    读取 raw_dir/*.gml，
    1. 用 Leiden 算法给每个节点打 community 属性；
    2. 构造 node features = [one-hot region_label; one-hot community]；
    3. 将 region_label 作为 y。
    """
    def __init__(self, root: str, resolution: float = 1.0,
                 transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)
        self.resolution = resolution

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in glob.glob(os.path.join(self.raw_dir, '*.gml'))]

    @property
    def processed_file_names(self):
        return ['processed_gcn.pt']

    def download(self):
        # 原始 .gml 已放在 raw_dir
        pass

    def process(self):
        data_list = []
        for fname in self.raw_file_names:
            path = os.path.join(self.raw_dir, fname)
            # 1. 读 GML
            G_nx = nx.read_gml(path)

            # 2. Leiden 分区
            G_ig = ig.Graph.from_networkx(G_nx)
            part = leidenalg.find_partition(
                G_ig,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution
            )
            membership = part.membership
            for node, com in zip(G_nx.nodes(), membership):
                G_nx.nodes[node]['community'] = com

            # 3. 转 PyG Data
            data = from_networkx(G_nx)

            # 节点标签 y
            y = torch.tensor(
                [G_nx.nodes[n]['region_label'] for n in G_nx.nodes()],
                dtype=torch.long
            )
            data.y = y

            # 4. 拼接特征
            # 原始 semantic
            sem_vals = [G_nx.nodes[n]['region_label'] for n in G_nx.nodes()]
            m = max(sem_vals) + 1
            # community
            com_vals = membership
            m2 = max(com_vals) + 1

            sem_feat = torch.zeros(data.num_nodes, m)
            com_feat = torch.zeros(data.num_nodes, m2)
            for i, (s, c) in enumerate(zip(sem_vals, com_vals)):
                sem_feat[i, s] = 1
                com_feat[i, c] = 1
            data.x = torch.cat([sem_feat, com_feat], dim=1)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])