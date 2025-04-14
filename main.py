from config_util import MP3D_DATASET_PATH,DEFAULT_SAVE_PATH
from adaptive_topo.topo import get_untopo_graph
from adaptive_topo.construct_graph import construct_topological_graph_based_scene
from sim_connect.hb import init_simulator

if __name__ == "__main__":
    test_scene = "17DRP5sb8fy"
    scene_path = f"{MP3D_DATASET_PATH}/{test_scene}/{test_scene}.glb"
    graph_save_path = f"{DEFAULT_SAVE_PATH}/graph"
    # Initialize Habitat-Sim
    sim = init_simulator(scene_path,True)

    # construct topological graph based on scene
    graph = construct_topological_graph_based_scene(sim,scene_path,save_path=graph_save_path,is_level_derive=True,show_graph = True,save_graph=True)
    
    # close the habitat sim
    sim.close()