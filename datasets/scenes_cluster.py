import os
import glob
import torch
import networkx as nx
import igraph as ig
import leidenalg
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import json

class ScenesGCNDataset(InMemoryDataset):
    """
    使用 region_id 图间一致 做标签 & 特征。
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
                 resolution: float = 1.0,
                 transform=None, pre_transform=None):    
        # collect all region_id, and leiden id
        raw_globs = glob.glob(os.path.join(root,'raw', '*.gml'))
        ids = set()
        max_com = 0
        for path in raw_globs:
            G_tmp = nx.read_graphml(path)
            for n,d in G_tmp.nodes(data=True):
                ids.add(int(d['region_id']))
            G_ig = ig.Graph.from_networkx(G_tmp)
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,seed=12345)
            max_com = max(max_com, len(set(part.membership)))
        self.max_com = max_com

        self.region_ids = sorted(ids)
        self.region_to_idx = {rid:i for i,rid in enumerate(self.region_ids)}
        self.num_regions = len(self.region_ids)

        # read object class
        with open(os.path.join(root,'raw', 'all_objects.json'),'r') as file:
            data = json.load(file)
            self.object_classes = data.get('objects')
        # 物体类别映射
        self.obj_to_idx = {o:i for i,o in enumerate(self.object_classes)}
        self.num_obj = len(self.object_classes)
        self.resolution = resolution
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            # Load processed if exists, allowing pickle of PyG objects
            data_file = self.processed_paths[0]
        if os.path.exists(data_file):
            try:
                self.data, self.slices = torch.load(data_file, weights_only=False)
            except TypeError:
                self.data, self.slices = torch.load(data_file)
            except Exception as e:
                print(f"Warning: failed to load processed dataset: {e}, reprocessing.")
                os.remove(data_file)

    @property
    def raw_file_names(self):
        files = glob.glob(os.path.join(self.raw_dir, '*.gml'))
        return [os.path.basename(p) for p in files]

    @property
    def processed_file_names(self):
        return ['processed_g.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for fname in self.raw_file_names:
            print(fname)
            path = os.path.join(self.raw_dir, fname)
            print(path)
            G_nx = nx.read_graphml(path)

            # Leiden
            G_ig = ig.Graph.from_networkx(G_nx)
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution,seed=12345)
            membership = part.membership
            
            for node, com in zip(G_nx.nodes(), membership):
                G_nx.nodes[node]['community'] = com
                if G_nx.nodes[node].get('objects',None) == None:
                    G_nx.nodes[node]['objects'] = ''
            
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
            com_feat = F.one_hot(torch.tensor(membership,dtype=torch.long), num_classes=self.max_com).float()
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