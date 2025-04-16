import os

import networkx as nx
import magnum as mn
from config_util import MP3D_DATASET_PATH,DEFAULT_SAVE_PATH
from sim_connect.hb import create_viewer


def add_vis_attributes_to_graph2(graph,glb_path,out_path):
    """
    :graph: the topological graph
    :out_path: the saved image path
    """
    # Add multi-view image information to the graph
    scene_path = graph.graph.get("scene")
    viewer = create_viewer(glb_path)

    scene_id = os.path.splitext(os.path.basename(scene_path))[0]
    out_dir = os.path.join(out_path, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    for node_id, data in graph.nodes(data=True):
        position = data["position"]
        # teleport the agent to the goal
        viewer.transit_to_goal(position)
        image_paths = {}
        # Save image for each direction
        # need 4 different angle rendering view image
        view_angles = {
            "front": 0,
            "right": -90,
            "back": 180,
            "left": 90
        }
        for view_name, yaw_deg in view_angles.items():
            agent_state = viewer.agent.get_state()
            agent_state.rotation = mn.Quaternion.rotation(
                mn.Deg(yaw_deg), mn.Vector3(0, 1, 0)
            )
            viewer.agent.set_state(agent_state)

            image_filename = f"{scene_id}_{node_id}_{view_name}.png"
            image_path = os.path.join(out_dir, image_filename)

            viewer.save_viewpoint_image(image_path)
            image_paths[view_name] = image_path
        graph.nodes[node_id]["image_paths"] = image_paths

    viewer.exec()
    return graph

if __name__ == "__main__":
    test_scene = "17DRP5sb8fy"
    scene_path = f"{MP3D_DATASET_PATH}/{test_scene}/{test_scene}.glb"
    graph_save_path = f"{DEFAULT_SAVE_PATH}/graph"
    # Initialize Habitat-Sim
    # sim = init_simulator(scene_path,True)

    # construct topological graph based on scene
    # graph = construct_topological_graph_based_scene(sim,scene_path,save_path=graph_save_path,is_level_derive=True,show_graph = True,save_graph=True)
    graph_file_path = f'{DEFAULT_SAVE_PATH}/graph/{test_scene}/17DRP5sb8fy_level0_navgraph.gml'
    # construct
    g = nx.read_graphml(graph_file_path)
    print(f"Graph has {len(g.nodes())} nodes and {len(g.edges())} edges.")
    save_image_path = f'{DEFAULT_SAVE_PATH}/img/'
    add_vis_attributes_to_graph2(g,scene_path,save_image_path)
