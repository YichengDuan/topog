from config_util import MP3D_DATASET_PATH
from adaptive_topo.topo import get_untopo_graph
from adaptive_topo.construct_graph import construct_topological_graph_based_scene
from sim_connect.hb import init_simulator

if __name__ == "__main__":
    test_scene = "17DRP5sb8fy"
    scene_path = f"{MP3D_DATASET_PATH}/{test_scene}/{test_scene}.glb"

    # Initialize Habitat-Sim
    sim = init_simulator(scene_path,True)

    # construct topological graph based on scene
    graph = construct_topological_graph_based_scene(sim,scene_path,show_graph = True)
    
    # close the habitat sim
    sim.close()