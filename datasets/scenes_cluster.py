import os, glob
import torch
import networkx as nx
import igraph as ig
import leidenalg

from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx

class ScenesClusterDataset(InMemoryDataset):
    """
    每个图代表一个场景，节点已有 region_label（语义真值）属性。
    我们先做 Leiden community，将 community 当作 pseudo‑feature；再用原始 region_label 作为节点 y。
    """
    def __init__(self, root: str, resolution: float=1.0, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # processed_paths[0] 对应文件 processed.pt
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.resolution = resolution

    @property
    def raw_file_names(self):
        # 假设 raw/ 目录下都是 .gpickle
        return [os.path.basename(p) for p in glob.glob(os.path.join(self.raw_dir, "*.gpickle"))]

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def download(self):
        # 如果你已经把所有 .gpickle 放在 raw_dir，这里可以留空
        pass

    def process(self):
        data_list = []
        for raw_name in self.raw_file_names:
            path = os.path.join(self.raw_dir, raw_name)
            # 1. 读取 NX 图
            G_nx = nx.read_gpickle(path)
            # 2. Leiden 分区
            G_ig = ig.Graph.from_networkx(G_nx)
            part = leidenalg.find_partition(
                G_ig,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution
            )
            membership = part.membership
            # 把 community 编号写回 NX 节点属性
            for node, com in zip(G_nx.nodes(), membership):
                G_nx.nodes[node]['community'] = com

            # 3. 转 PyG 的 Data
            data = from_networkx(G_nx)
            # 假设原始 region_label 存在 node_attr region_label
            # 把它转成 tensor
            y = torch.tensor(
                [ G_nx.nodes[i]['region_label'] for i in range(data.num_nodes) ],
                dtype=torch.long
            )
            data.y = y

            # 4. 构造节点特征 x = [原始 one‑hot 语义 + community one‑hot]
            semantic_vals = [G_nx.nodes[i]['region_label'] for i in range(data.num_nodes)]
            m = max(semantic_vals) + 1
            com_vals = membership
            m2 = max(com_vals) + 1

            sem_feat = torch.zeros(data.num_nodes, m)
            com_feat = torch.zeros(data.num_nodes, m2)
            for i, (s, c) in enumerate(zip(semantic_vals, com_vals)):
                sem_feat[i, s] = 1
                com_feat[i, c] = 1
            data.x = torch.cat([sem_feat, com_feat], dim=1)

            data_list.append(data)

        # 批量 save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])