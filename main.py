from config_util import MP3D_DATASET_PATH
from adaptive_topo.topo import init_simulator, get_untopo_graph


if __name__ == "__main__":
    scene_path = f"{MP3D_DATASET_PATH}/17DRP5sb8fy/17DRP5sb8fy.glb"

    # TESTING 1 : find all possable navpoints on one scene
    # Initialize Habitat-Sim
    sim = init_simulator(scene_path)
    # Get the top-down view and navpoints
    output_path = "./results/17DRP5sb8fy_navpoints_topdown.png"
    get_untopo_graph(sim, output_path)
    