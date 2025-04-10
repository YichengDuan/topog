from config_util import MP3D_DATASET_PATH
from adaptive_topo.topo import get_untopo_graph
from sim_connect.hb import init_simulator

if __name__ == "__main__":
    test_scene = "17DRP5sb8fy"
    scene_path = f"{MP3D_DATASET_PATH}/{test_scene}/{test_scene}.glb"

    # TESTING 1 : find all possable navpoints on one scene
    # Initialize Habitat-Sim
    sim = init_simulator(scene_path)
    # Get the top-down view and navpoints
    output_path = f"./results/{test_scene}_navpoints_topdown.png"
    get_untopo_graph(sim, output_path,resolution=0.2, yflip=False, semantic_overlay=False)
    # Get the top-down view and navpoints with semantic overlay
    output_path = f"./results/{test_scene}_navpoints_topdown_semantic.png"
    get_untopo_graph(sim, output_path, resolution=0.2, yflip=False, semantic_overlay=True)
    

    sim.close()