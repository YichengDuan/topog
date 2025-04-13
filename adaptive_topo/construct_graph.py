import itertools
import os

import habitat_sim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import magnum as mn

from adaptive_topo.graph_util import estimate_num_nodes, sampling_nav, add_edge

def construct_topological_graph_based_scene(sim, scene_path ,save_path = None ,is_level_derive = False , save_graph = False, show_graph = False):
    """
    Construct the topological graph based on the 3d scene(glb. file)

    :param scene_path: the path to the glb file
    :param save_path: the path to save the gml. file
    :param is_level_derive: the variable determining construct graph for each level instead of the whole scene
    :param sim: the habitat_sim object
    :param show_graph: whether show the graph or not
    :param save_graph: whether save the graph or not
    """
    # get the id of the scene
    scene_id = os.path.basename(os.path.dirname(scene_path))

    # semantic scene label for the graph
    semantic_scene = sim.semantic_scene

    # get the pathfinder
    pathfinder = sim.pathfinder

    # define the number of the nodes N
    N = estimate_num_nodes(pathfinder)
    # sampling the nodes from navmesh
    nodes = sampling_nav(2,N,pathfinder,sim)

    # create the graph
    graph = nx.Graph()
    # label the graph a scene label
    graph.graph["scene"] = scene_id

    # add attributed node to the graph
    for i, point in enumerate(nodes):
        region_id, region_name,level = manual_region_lookup(point, semantic_scene)
        graph.add_node(
            i,
            position=point,
            region_id=region_id,
            region_name=region_name,
            level = level
        )
    # add edges
    graph = add_edge(pathfinder,graph,nodes)

    # print the information of the graph
    print(f"Topological Graph for scene[{scene_id}] constructed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    # print the node positions
    for node_id, data in graph.nodes(data=True):
        print(f"Node {node_id} → position: {data['position']}")

    # ---------------------SHOW GRAPH---------------------------
    if show_graph:
        showing_graph(graph)
    # ------------------- save -------------------------
    if save_graph:
        # save the graph to a file if a path is provided
        # convert the position attribute into a string for safety writing
        for i, data in graph.nodes(data=True):
            pos = data['position']
            data['position'] = ",".join(f"{v}" for v in pos)
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            nx.write_graphml(graph,f"{save_path}/{scene_id}_navgraph.gml")

    return graph

def showing_graph(graph):
    matplotlib.use('TkAgg')
    # Extract region labels
    region_labels = nx.get_node_attributes(graph, "region_name")
    unique_labels = list(set(region_labels.values()))
    # Assign shapes and colors
    shapes = ['o', 's', '^', 'D', 'v', 'h', '*', 'p', 'X', '<', '>']
    colors = cm.get_cmap('tab20', len(unique_labels))(np.arange(len(unique_labels)))
    label_to_shape = dict(zip(unique_labels, itertools.cycle(shapes)))
    label_to_color = dict(zip(unique_labels, colors))
    # Group nodes by label
    nodes_by_label = {}
    for node, data in graph.nodes(data=True):
        label = data.get("region_name", "unknown")
        nodes_by_label.setdefault(label, []).append(node)
    # 2D positions for plotting
    pos_2d = {i: (p[0], -p[2]) for i, p in nx.get_node_attributes(graph, 'position').items()}
    plt.figure(figsize=(12, 10))
    # Draw edges first
    nx.draw_networkx_edges(graph, pos=pos_2d, edge_color='gray')
    # Draw nodes by label (shape + color)
    for label, nodes in nodes_by_label.items():
        shape = label_to_shape[label]
        color = label_to_color[label]
        nx.draw_networkx_nodes(
            graph,
            pos=pos_2d,
            nodelist=nodes,
            node_shape=shape,
            node_color=[color],
            node_size=80,
            label=label
        )
    plt.legend(title="Region")
    plt.title("Semantic NavGraph with Region Shapes & Colors")
    plt.axis("equal")
    plt.grid(False)
    plt.show()

def manual_region_lookup(point, semantic_scene, margin = 0.0 ,y_margin=0.25):
    for idx, region in enumerate(semantic_scene.regions):
        aabb = region.aabb
        level = region.level.id
        if (aabb.min.x - margin <= point.x <= aabb.max.x + margin and
                aabb.min.z - margin <= point.z <= aabb.max.z + margin and
                aabb.min.y - y_margin <= point.y <= aabb.max.y + y_margin):
            return idx, region.category.name(),level
    return 999999, "unknown", "unknown"

def add_vis_attributes_to_graph(graph,out_path):
    """
    :graph: the topological graph
    :out_path: the saved image path
    """
    # Add multi-view image information to the graph
    scene_path = graph.graph.get("scene")
    viewer = create_viewer(scene_path)

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
