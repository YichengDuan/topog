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
    使用 region_id（图间一致）做标签 & 特征。
    Node attrs in GraphML:
      - 'position': "x,y,z"
      - 'region_id': int, 全局一致
      - 'objects': comma-separated names
    Args:
      root: dataset 目录, 含 raw/ 和 processed/
      object_classes: 全局物体类别列表
      resolution: Leiden 分辨率
    """
    def __init__(self, root: str,
                 object_classes: list[str],
                 resolution: float = 1.0,
                 transform=None, pre_transform=None):
        # 收集所有 region_id
        raw_globs = glob.glob(os.path.join(root, 'raw', '*.graphml')) + \
                    glob.glob(os.path.join(root, 'raw', '*.gml'))
        ids = set()
        for path in raw_globs:
            G_tmp = nx.read_graphml(path) if path.endswith('.graphml') else nx.read_gml(path)
            for n,d in G_tmp.nodes(data=True):
                ids.add(int(d['region_id']))
        self.region_ids = sorted(ids)
        self.region_to_idx = {rid:i for i,rid in enumerate(self.region_ids)}
        self.num_regions = len(self.region_ids)

        # 物体类别映射
        self.obj_to_idx = {o:i for i,o in enumerate(object_classes)}
        self.num_obj = len(object_classes)
        self.resolution = resolution
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = glob.glob(os.path.join(self.raw_dir, '*.graphml')) + \
                glob.glob(os.path.join(self.raw_dir, '*.gml'))
        return [os.path.basename(p) for p in files]

    @property
    def processed_file_names(self):
        return ['processed_gcn.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for fname in self.raw_file_names:
            path = os.path.join(self.raw_dir, fname)
            G_nx = nx.read_graphml(path) if path.endswith('.graphml') else nx.read_gml(path)

            # Leiden
            G_ig = ig.Graph.from_networkx(G_nx)
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution)
            membership = part.membership
            for node, com in zip(G_nx.nodes(), membership):
                G_nx.nodes[node]['community'] = com

            # 转 PyG
            data = from_networkx(G_nx)

            # y = region_id idx
            rids = [int(G_nx.nodes[n]['region_id']) for n in G_nx.nodes()]
            y = torch.tensor([self.region_to_idx[r] for r in rids], dtype=torch.long)
            data.y = y

            # x 特征拼接
            coords = torch.tensor([
                list(map(float, G_nx.nodes[n]['position'].split(',')))
                for n in G_nx.nodes()], dtype=torch.float)
            # region one-hot
            sem_feat = F.one_hot(y, num_classes=self.num_regions).float()
            # community one-hot
            com_feat = F.one_hot(torch.tensor(membership), num_classes=max(membership)+1).float()
            # objects hist
            obj_hist = []
            for n in G_nx.nodes():
                objs = G_nx.nodes[n].get('objects','')
                names = [s.strip() for s in objs.split(',') if s.strip()]
                hist = torch.zeros(self.num_obj)
                for nm in names:
                    idx = self.obj_to_idx.get(nm)
                    if idx is not None: hist[idx]+=1
                if hist.sum()>0: hist /= hist.sum()
                obj_hist.append(hist)
            obj_hist = torch.stack(obj_hist, dim=0)
            data.x = torch.cat([coords, sem_feat, com_feat, obj_hist], dim=1)

            # 边特征
            row, col = data.edge_index
            weights = []
            for u,v in zip(row.tolist(), col.tolist()):
                w = G_nx.edges[list(G_nx.nodes())[u], list(G_nx.nodes())[v]].get('weight')
                if w is None:
                    w = torch.norm(coords[u]-coords[v]).item()
                weights.append(w)
            data.edge_attr = torch.tensor(weights, dtype=torch.float).view(-1,1)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])