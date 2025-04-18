import os
import glob
import torch
import networkx as nx
import igraph as ig
import leidenalg
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

class ScenesGCNDataset(InMemoryDataset):
    """
    Dataset for multiple scenes stored as .gml files.
    Each scene graph:
      - node attributes:
          'coord': tuple(x, y, z)
          'region_label': int semantic label
          'obj_hist': list or vector of object class histogram
      - We add 'community' via Leiden
      - x = [coord; one-hot(region_label); one-hot(community); obj_hist]
      - edge_attr = euclidean distance between coords
      - y = region_label (node-level)
    """
    def __init__(self, root: str, resolution: float = 1.0,
                 transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.resolution = resolution
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(p)
                for p in glob.glob(os.path.join(self.raw_dir, '*.gml'))]

    @property
    def processed_file_names(self):
        return ['processed_gcn.pt']

    def download(self):
        # .gml 已放在 raw_dir
        pass

    def process(self):
        data_list = []
        for fname in self.raw_file_names:
            path = os.path.join(self.raw_dir, fname)
            # 1. 读取 .gml 图
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

            # 4. 构造节点标签 y
            y = torch.tensor(
                [G_nx.nodes[n]['region_label'] for n in G_nx.nodes()],
                dtype=torch.long
            )
            data.y = y

            # 5. 构造节点特征 x
            # 5.1: 坐标
            coords = torch.tensor(
                [G_nx.nodes[n]['coord'] for n in G_nx.nodes()],
                dtype=torch.float
            )  # [N,3]
            # 5.2: 语义 label one-hot
            sem = [G_nx.nodes[n]['region_label'] for n in G_nx.nodes()]
            m_sem = max(sem) + 1
            sem_feat = F.one_hot(torch.tensor(sem), num_classes=m_sem).float()  # [N, m_sem]
            # 5.3: community one-hot
            com = membership
            m_com = max(com) + 1
            com_feat = F.one_hot(torch.tensor(com), num_classes=m_com).float()  # [N, m_com]
            # 5.4: object histogram
            obj_hist = torch.stack(
                [torch.tensor(G_nx.nodes[n]['obj_hist'], dtype=torch.float)
                 for n in G_nx.nodes()], dim=0
            )  # [N, num_obj_classes]

            data.x = torch.cat([coords, sem_feat, com_feat, obj_hist], dim=1)

            # 6. 构造边特征 edge_attr = distance
            row, col = data.edge_index
            edge_attr = torch.norm(coords[row] - coords[col], dim=1, keepdim=True)
            data.edge_attr = edge_attr

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
