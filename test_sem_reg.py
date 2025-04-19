from adaptive_topo.topo import get_untopo_graph
from config_util import MP3D_DATASET_PATH
from sim_connect.hb import init_simulator

test_id = "17DRP5sb8fy"
file_path = f"{MP3D_DATASET_PATH}/{test_id}/{test_id}.glb"

save_path = './results/'
img_name = f"{test_id}_navpoints_topdown.png"

sim = init_simulator(file_path)

get_untopo_graph(sim=sim, output_path=f"{save_path}{img_name}",semantic_overlay = False)
img_name = f"{test_id}_navpoints_topdown_semantic.png"
get_untopo_graph(sim=sim, output_path=f"{save_path}{img_name}",semantic_overlay = False)

sim.close()