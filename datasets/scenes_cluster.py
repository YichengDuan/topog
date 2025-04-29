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
import tqdm


class ScenesGCNDataset(InMemoryDataset):
    """
    Node attrs in GraphML:
      - 'position': "x,y,z"
      - 'region_id': int 
      - 'objects': comma-separated names
    Args:
      root: dataset, must include raw/ and processed/
      object_classes: object classes list of strings
      resolution: Leiden resolution
    """
    def __init__(self, root: str,
                 resolution: float = 1.0,
                 transform=None, pre_transform=None, config: dict = None):    
        
        # Default config: all features enabled
        default_cfg = {'position':1, 'community':1, 'objects':1, 'edge_weight':1}
        self.config = default_cfg if config is None else {**default_cfg, **config}
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

        # Build a unique filename for processed data
        flags = [f"{k}{self.config[k]}" for k in ['position','community','objects','edge_weight']]
        self._processed_file = f"processed_{'_'.join(flags)}.pt"

        super().__init__(root, transform, pre_transform)
        
        # Attempt to load if exists
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):
        files = glob.glob(os.path.join(self.raw_dir, '*.gml'))
        return [os.path.basename(p) for p in files]

    @property
    def processed_file_names(self):
        return [self._processed_file]

    def download(self):
        pass

    def process(self):
        data_list = []
        for fname in tqdm.tqdm(self.raw_file_names,desc=f"building dataset{self._processed_file}"):
            path = os.path.join(self.raw_dir, fname)
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
            node_list = list(G_nx.nodes())
            # y = region_id idx
            rids = [int(G_nx.nodes[n]['region_id']) for n in G_nx.nodes()]
            y = torch.tensor([self.region_to_idx[r] for r in rids], dtype=torch.long)
            data.y = y

            # Node features
            feat_list = []
            if self.config['position']:
                # position feature
                coords = torch.tensor([
                    list(map(float, G_nx.nodes[n]['position'].split(',')))
                    for n in node_list], dtype=torch.float)
                feat_list.append(coords)
            if self.config['community']:
                # community one-hot
                com_feat = F.one_hot(torch.tensor(membership,dtype=torch.long), num_classes=self.max_com).float()
                feat_list.append(com_feat)
            if self.config['objects']:
                # objects hist
                obj_hist = []
                for n in node_list:
                    objs = G_nx.nodes[n].get('objects','')
                    names = [s.strip() for s in objs.split(',') if s.strip()]
                    hist = torch.zeros(self.num_obj)
                    for nm in names:
                        idx = self.obj_to_idx.get(nm)
                        if idx is not None: hist[idx]+=1
                    if hist.sum()>0: hist /= hist.sum()
                    obj_hist.append(hist)
                obj_hist = torch.stack(obj_hist, dim=0)
                feat_list.append(obj_hist)

            if feat_list:
                data.x = torch.cat(feat_list, dim=1)
            else:
                data.x = torch.ones((len(node_list),1), dtype=torch.float)
            # data.x = torch.cat([coords, com_feat, obj_hist], dim=1)


            # 边特征
            if self.config['edge_weight']:
                row, col = data.edge_index
                weights = []
                for u,v in zip(row.tolist(), col.tolist()):
                    w = G_nx.edges[list(G_nx.nodes())[u], list(G_nx.nodes())[v]].get('weight')
                    if w is None:
                        w = torch.norm(coords[u]-coords[v]).item()
                    weights.append(w)
                data.edge_attr = torch.tensor(weights, dtype=torch.float).view(-1,1)
            else:
                # No edge weights: use constant 1 for all edges
                num_edges = data.edge_index.size(1)
                data.edge_attr = torch.ones((num_edges, 1), dtype=torch.float)
            
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])