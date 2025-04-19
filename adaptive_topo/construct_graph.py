import os
import itertools
import magnum as mn
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from adaptive_topo.graph_util import (
    estimate_num_nodes,
    sampling_nav,
    add_edge_ray,
    manual_region_lookup
)
from sim_connect.hb import create_viewer, init_simulator
import habitat_sim
from config_util import MP3D_DATASET_PATH
from vision_gd import ObjectExtractor
import tqdm
class GraphConstructor:
    """
    Constructs a topological graph from a 3D Habitat-Sim scene.
    """

    def __init__(
        self,
        scene_id,
        save_gml_path=None,
        save_img_path=None,
        is_level_derive=False,
        save_graph=False,
        show_graph=False,
        object_extractor:ObjectExtractor= None,
    ):
        self.scene_id = scene_id
        self.scene_path = os.path.join(MP3D_DATASET_PATH, scene_id, f"{scene_id}.glb")
        self.save_gml_path = save_gml_path
        self.save_img_path = save_img_path
        self.is_level_derive = is_level_derive
        self.save_graph = save_graph
        self.show_graph = show_graph
        self.sim = init_simulator(self.scene_path,is_physics=True)
        self.semantic_scene = self.sim.semantic_scene
        self.pathfinder = self.sim.pathfinder
        self.object_extractor = object_extractor
        self.object_list = []
        os.makedirs(self.save_gml_path, exist_ok=True)
        os.makedirs(self.save_img_path, exist_ok=True)

    def construct_topological_graph(self):
        """
        Main entry: construct either per-level or whole-scene graph(s).
        Returns a single graph or list of graphs.
        """
        # sample nav points
        N = estimate_num_nodes(self.pathfinder)
        samples = sampling_nav(2, N, self.pathfinder, self.sim)
        # enrich samples with semantic info
        nodes_info = []
        for s in samples:
            region_id, region_name, level = manual_region_lookup(
                s['point'], self.semantic_scene
            )
            nodes_info.append({
                'point': s['point'],
                'radius': s['radius'],
                'region_id': region_id,
                'region_name': region_name,
                'level': level,
            })

        if self.is_level_derive:
            return self._construct_per_level(nodes_info)
        return self._construct_whole_scene(nodes_info)

    def _construct_per_level(self, nodes_info):
        graph_list = []
        for lvl in self.semantic_scene.levels:
            lvl_id = lvl.id
            lvl_nodes = [n for n in nodes_info if n['level'] == lvl_id]
            if not lvl_nodes:
                continue
            G = nx.Graph()
            G.graph.update({
                'scene': f"{self.scene_id}_level{lvl_id}",
                'scene_path': self.scene_path
            })
            for idx, info in enumerate(lvl_nodes):
                G.add_node(
                    idx,
                    node_type='topology',
                    position=info['point'],
                    region_id=info['region_id'],
                    region_name=info['region_name'],
                    level=info['level'],
                )
            G = add_edge_ray(self.pathfinder, G, lvl_nodes)
            # only remain the largest componet
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            
            fig = self._plot_graph(G)
            graph_list.append({'level': lvl_id, 'graph': G, 'fig': fig})

            if self.show_graph:
                fig.show()

        if self.save_graph and self.save_gml_path:
            for item in graph_list:
                lvl, G, fig = item['level'], item['graph'], item['fig']
                
                self._prepare_and_save(G, fig, lvl)
        return graph_list

    def _construct_whole_scene(self, nodes_info):
        G = nx.Graph()
        G.graph.update({'scene': self.scene_id, 'scene_path': self.scene_path})
        for idx, info in enumerate(nodes_info):
            G.add_node(
                idx,
                node_type='topology',
                position=info['point'],
                region_id=info['region_id'],
                region_name=info['region_name'],
                level=info['level'],
            )
        G = add_edge_ray(self.pathfinder, G, nodes_info)
        print(f"Constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if self.show_graph:
            fig = self._plot_graph(G)
            fig.show()
        if self.save_graph and self.save_gml_path:
            self._prepare_and_save(G, None, self.save_gml_path)
        return G

    def _plot_graph(self, G):
        # semantic labels → shapes & colors
        labels = nx.get_node_attributes(G, 'region_name')
        unique = list(set(labels.values()))
        shapes = itertools.cycle(['o','s','^','D','v','h','*','p','X','<','>'])
        cmap = cm.get_cmap('tab20', len(unique))
        label2shape = dict(zip(unique, shapes))
        label2color = dict(zip(unique, cmap(range(len(unique)))))

        pos2d = {n: (p[0], -p[2]) for n,p in nx.get_node_attributes(G,'position').items()}
        fig, ax = plt.subplots(figsize=(12,10))
        nx.draw_networkx_edges(G, pos2d, edge_color='gray', ax=ax)
        for lbl in unique:
            nodes = [n for n,v in nx.get_node_attributes(G,'region_name').items() if v==lbl]
            nx.draw_networkx_nodes(
                G, pos2d,
                nodelist=nodes,
                node_shape=label2shape[lbl],
                node_color=[label2color[lbl]],
                node_size=80,
                label=lbl,
                ax=ax
            )
        ax.legend(title='Region')
        ax.axis('equal'); ax.grid(False)
        return fig

    def _prepare_and_save(self, G:nx.Graph, fig, level=None):
        # serialize node positions
        for _,d in G.nodes(data=True):
            d['position'] = ",".join(map(str,d['position']))
        # add visual information/ object seen from the node in to one node attr
        self._add_vis_attributes_to_graph(G)
        # save GML
        fname = f"{self.scene_id}"
        if level is not None:
            fname += f"_level{level}"
        gml_path = os.path.join(self.save_gml_path, f"{fname}_navgraph.gml")
        nx.write_graphml(G, gml_path)
        print(f"Saved graph: {gml_path}")

        if fig is not None:
            img_path = os.path.join(self.save_img_path, f"{fname}_navgraph.png")
            fig.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot: {img_path}")

       

    def _add_vis_attributes_to_graph(self, G):
        """
        Add visualization attributes to the graph. 
        Return an graph with seen objects histgram as node attrs.
        
        """

        viewer = create_viewer(self.scene_path)
        
        for nid, data in tqdm.tqdm(G.nodes(data=True)):
            pos = data['position']
            if isinstance(pos, str):
                pos = list(map(float, pos.split(',')))
            vec = mn.Vector3(*pos)
            viewer.transit_to_goal(vec)
            node_image_list = []
            for i in range(4):
                current_view_img = viewer.get_viewpoint_img(drop_depth=True)
                node_image_list.append(current_view_img)
                viewer.move_and_look('turn_right', steps=int(90/1.5))
            objects = self.object_extractor.extract_batch(node_image_list)
            # save objects
            objects = list(set(objects))
            self.object_list.extend(objects)
            data['objects'] = ",".join(objects)
        
        viewer.close()

    def get_all_objects(self):
        """
        Get all objects in the scene.
        """
        self.object_list = list(set(self.object_list))
   
        return self.object_list