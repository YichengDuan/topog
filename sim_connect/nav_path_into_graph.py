import os

import argparse
import habitat_sim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib

from hb import create_viewer

matplotlib.use('TkAgg')


def estimate_num_nodes(pathfinder, avg_spacing=0.5):
    """
    Estimate number of nodes by sampling and estimating navigable area.
    And the number N is defined by N = A/(pi * radius^2), where A denotes the area

    :param pathfinder: Habitat-Sim pathfinder object
    :param avg_spacing: desired spacing between nodes (m)
    :return: estimated number of nodes
    """
    nav_area = pathfinder.navigable_area
    estimated_n = int(nav_area / (np.pi * (avg_spacing ** 2)))

    print(f"[Estimate] ~{nav_area:.2f} m² → Estimated nodes: {estimated_n}")
    return estimated_n


def point_on_path(pt, path_points, tolerance=0.5):
    """
    :param pt:  candidate node
    :param path_points: the result of path.points from find_path()
    :param tolerance:
    :return: if there is a node in the middle of the path
    """
    pt = np.array(pt)
    for i in range(len(path_points) - 1):
        seg_start = np.array(path_points[i])
        seg_end = np.array(path_points[i + 1])
        vec = seg_end - seg_start
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        proj = np.dot(pt - seg_start, vec_norm)
        closest = seg_start + proj * vec_norm
        dist = np.linalg.norm(pt - closest)

        # Check if projected point is between segment endpoints and within tolerance
        if dist < tolerance:
            if 0 <= proj <= np.linalg.norm(seg_end - seg_start):
                return True
    return False

def adaptive_radius(p, pathfinder, alpha=0.8, r_min=0.25, r_max=2.5):
    d = pathfinder.distance_to_closest_obstacle(p)
    return np.clip(alpha * d, r_min, r_max)


def adaptive_poisson_sample(pathfinder, num_nodes, alpha=0.8, r_min=0.5, r_max=2.5, max_attempts_per_point=30):
    """
    Sample N adaptive Poisson disk points on the navmesh.

    :param pathfinder: habitat_sim.PathFinder object
    :param num_nodes: number of nodes to sample
    :param alpha: scale factor for adaptive radius
    :param r_min: minimum radius
    :param r_max: maximum radius
    :param max_attempts_per_point: attempts per point before giving up
    :return: list of sampled navigable 3D points
    """
    samples = []
    attempts = 0
    max_total_attempts = num_nodes * max_attempts_per_point

    while len(samples) < num_nodes and attempts < max_total_attempts:
        p = pathfinder.get_random_navigable_point()
        r_p = adaptive_radius(p, pathfinder, alpha, r_min, r_max)

        # Accept only if far from all previous points
        too_close = False
        for s in samples:
            dist = np.linalg.norm(np.array(p) - np.array(s))
            r_s = adaptive_radius(s, pathfinder, alpha, r_min, r_max)
            if dist < min(r_p, r_s):
                too_close = True
                break

        if not too_close:
            samples.append(p)

        attempts += 1

    print(f"[Sampling] Sampled {len(samples)} points after {attempts} attempts.")
    return samples

def create_graph_based_scene(scene_path, config_path,distance_threshold = 3.0,block_tolerance = 0.5, save_path = None):
    # get the id of the scene
    scene_id = os.path.basename(os.path.dirname(scene_path))

    # Basic simulator config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False  # no physics needed
    # configuration file
    sim_cfg.scene_dataset_config_file = config_path

    # Create the simulator
    agent_cfg = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    # get the pathfinder
    pathfinder = sim.pathfinder

    # define the number of the nodes N
    N = estimate_num_nodes(pathfinder)
    # poisson
    # nodes = poisson_disk_sample(pathfinder, radius=1.0, max_samples= N)
    nodes = adaptive_poisson_sample(pathfinder, N)

    # create the graph
    graph = nx.Graph()
    # label the graph a scene label
    graph.graph["scene"] = scene_id

    for i, point in enumerate(nodes):
        graph.add_node(i, position=point)# node = index, attribute = position (x,y,z)

    # For each pair of nodes within distance threshold, check if there's a path
    # If so, add an edge with geodesic (shortest-path) distance as the edge weight

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            path = habitat_sim.ShortestPath()
            path.requested_start = nodes[i]
            path.requested_end = nodes[j]

            if sim.pathfinder.find_path(path):
                if path.geodesic_distance < distance_threshold:
                    # Now check if any OTHER node lies ON this path
                    blocked = False
                    for k in range(len(nodes)):
                        if k != i and k != j:
                            b = nodes[k]
                            if point_on_path(b, path.points, tolerance=block_tolerance):  # function defined below
                                blocked = True
                                break
                    if not blocked:
                        graph.add_edge(i, j, weight=path.geodesic_distance)

    # For example, show number of nodes/edges
    print(f"Graph constructed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # You can now export this graph or use NetworkX for shortest paths, visualization, etc
    pos_2d = {i: (p[0], p[2]) for i, p in nx.get_node_attributes(graph, 'position').items()}

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos=pos_2d,
        node_size=50,
        node_color='skyblue',
        edge_color='gray',
        labels={i: str(i) for i in graph.nodes()},
        font_size=8,
        with_labels=True,
    )

    plt.title("NavGraph Projection (XZ Plane)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    # print the node positions
    for node_id, data in graph.nodes(data=True):
        print(f"Node {node_id} → position: {data['position']}")

    # save the graph to a file if a path is provided
    # convert the position attribute into a string for safety writing
    for i, data in graph.nodes(data=True):
        pos = data['position']
        data['position'] = ",".join(f"{v}" for v in pos)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        nx.write_graphml(graph,f"{save_path}/{scene_id}_navgraph.gml")

    # close the sim
    sim.close()
    return graph



def add_attributes_to_graph(graph):
    # add image information to the graph
    scene_path = graph.graph.get("scene")
    viewer = create_viewer(scene_path)

    scene_id = os.path.splitext(os.path.basename(scene_path))[0]
    # Create output directory if it doesn't exist
    out_dir = os.path.join("../data/out", scene_id)
    os.makedirs(out_dir, exist_ok=True)

    for node_id, data in graph.nodes(data=True):
        position = data["position"]
        #Render the image at the current position and save it
        viewer.transit_to_goal(position)
        # Format the filename safely (e.g., x_y_z)
        pos_str = "_".join([f"{p:.2f}" for p in position])
        image_path = os.path.join(out_dir, f"{pos_str}.png")

        viewer.save_viewpoint_image(image_path)
        graph.nodes[node_id]["image_path"] = image_path
    #Start the application event loop (runs on the main thread).
    viewer.exec()



def main():
    parser = argparse.ArgumentParser(description="Construct a topological navgraph from a Habitat scene.")
    parser.add_argument(
        "--scene",
        type=str,
        default="../data/scene_datasets/mp3d/17DRP5sb8fy",
        help="Path to the folder containing the .glb and .navmesh files (e.g. ../data/scene_datasets/mp3d/17DRP5sb8fy)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../data/scene_datasets/mp3d/mp3d.scene_dataset_config.json",
        help="Path to the scene_dataset_config.json (default: mp3d config)",
    )
    args = parser.parse_args()

    # Automatically resolve file paths
    scene_dir = args.scene
    scene_id = os.path.join(scene_dir, os.path.basename(scene_dir) + ".glb")

    # Sanity check
    if not os.path.exists(scene_id):
        raise FileNotFoundError(f"Scene file not found: {scene_id}")

    # Call main graph function
    g = create_graph_based_scene(scene_id, args.config,save_path="../data/out/graph")

if __name__ == "__main__":
    main()