import networkx as nx

from config_util import MP3D_DATASET_PATH,DEFAULT_SAVE_PATH
# from adaptive_topo.topo import get_untopo_graph
from adaptive_topo.construct_graph import construct_topological_graph_based_scene, add_vis_attributes_to_graph
from sim_connect.hb import init_simulator

if __name__ == "__main__":
    test_scene = "17DRP5sb8fy"
    scene_path = f"{MP3D_DATASET_PATH}/{test_scene}/{test_scene}.glb"
    graph_save_path = f"{DEFAULT_SAVE_PATH}/graph"
    # Initialize Habitat-Sim
    sim = init_simulator(scene_path,True)

    # construct topological graph based on scene
    graph = construct_topological_graph_based_scene(sim,scene_path,save_path=graph_save_path,is_level_derive=True,show_graph = False,save_graph=True)
    # graph_file_path = f'{DEFAULT_SAVE_PATH}/graph/{test_scene}/17DRP5sb8fy_level0_navgraph.gml'
    # # # construct
    # g = nx.read_graphml(graph_file_path)
    # # print(f"Graph has {len(g.nodes())} nodes and {len(g.edges())} edges.")
    # save_image_path = f'{DEFAULT_SAVE_PATH}/img/'
    # close the habitat sim
    sim.close()