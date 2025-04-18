import os
import time

import pandas as pd

from config_util import MP3D_DATASET_PATH,DEFAULT_SAVE_PATH,MP3D_DATASET_SCENE_IDS_LIST
# from adaptive_topo.topo import get_untopo_graph
from adaptive_topo.construct_graph import construct_topological_graph_based_scene, add_vis_attributes_to_graph
from sim_connect.hb import init_simulator

if __name__ == "__main__":
    # test_scene = "17DRP5sb8fy"
    # scene_path = f"{MP3D_DATASET_PATH}/{test_scene}/{test_scene}.glb"
    # graph_save_path = f"{DEFAULT_SAVE_PATH}/graph"
    # # Initialize Habitat-Sim
    # sim = init_simulator(scene_path,True)
    #
    # # construct topological graph based on scene
    # graph = construct_topological_graph_based_scene(sim,scene_path,save_path=graph_save_path,is_level_derive=True,show_graph = False,save_graph=True)
    # # graph_file_path = f'{DEFAULT_SAVE_PATH}/graph/{test_scene}/17DRP5sb8fy_level0_navgraph.gml'
    # # # # construct
    # # g = nx.read_graphml(graph_file_path)
    # # # print(f"Graph has {len(g.nodes())} nodes and {len(g.edges())} edges.")
    # # save_image_path = f'{DEFAULT_SAVE_PATH}/img/'
    # # close the habitat sim
    # sim.close()

    time_list = []
    for scene_id in sorted(MP3D_DATASET_SCENE_IDS_LIST):
        scene_graph_root = os.path.join(DEFAULT_SAVE_PATH, "graph", scene_id)
        if os.path.exists(scene_graph_root):
            print(f"[Skip] Scene {scene_id} already has output folder → skipping.")
            continue
        print(f"\n=== Processing scene: {scene_id} ===")
        start_time = time.time()
        try:
            scene_path = os.path.join(MP3D_DATASET_PATH, scene_id, f"{scene_id}.glb")
            graph_save_path = os.path.join(DEFAULT_SAVE_PATH, "graph")
            sim = init_simulator(scene_path, True)

            construct_topological_graph_based_scene(
                sim,
                scene_path,
                save_path=graph_save_path,
                is_level_derive=True,
                show_graph=False,
                save_graph=True
            )

            sim.close()
        except Exception as e:
            print(f"[Error] Failed to process scene {scene_id}: {e}")
        elapsed = time.time() - start_time
        time_info = {"scene_id":scene_id,"time":elapsed}
        time_list.append(time_info)

    print("\n\n=== Scene Processing Time Summary ===")
    df = pd.DataFrame(time_list)
    print(df.to_string(index=False))

    total_time = df["time_sec"].sum()
    print(f"\n=== Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes) ===")