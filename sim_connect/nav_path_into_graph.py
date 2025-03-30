import argparse
import os

import habitat_sim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')


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


def poisson_disk_sample(pathfinder, radius=1.0, max_samples=200):
    samples = []
    attempts = 0
    max_attempts = max_samples * 10
    while len(samples) < max_samples and attempts < max_attempts:
        pt = pathfinder.get_random_navigable_point()
        if all(np.linalg.norm(np.array(pt) - np.array(p)) > radius for p in samples):
            samples.append(pt)
        attempts += 1
    return samples


def create_graph_based_scene(scene_id, config_path,distance_threshold = 3.0,block_tolerance = 0.5):
    # Basic simulator config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = False  # no physics needed
    # configuration file
    sim_cfg.scene_dataset_config_file = config_path

    # Create the simulator
    agent_cfg = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    # N = 200  # number of graph nodes
    # nodes = [sim.pathfinder.get_random_navigable_point() for _ in range(N)]
    # poisson
    nodes = poisson_disk_sample(sim.pathfinder, radius=1.0, max_samples=200)

    graph = nx.Graph()

    for i, point in enumerate(nodes):
        graph.add_node(i, position=point)  # node = index, attribute = position (x,y,z)

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
        with_labels=False
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
    return graph

def main():
    parser = argparse.ArgumentParser(description="Construct a topological navgraph from a Habitat scene.")
    parser.add_argument(
        "--scene",
        type=str,
        default="../data/scene_datasets/mp3d/17DRP5sb8fy",
        help="Path to the folder containing the .glb and .navmesh files (e.g. data/scene_datasets/mp3d/17DRP5sb8fy)",
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
    create_graph_based_scene(scene_id, args.config)

if __name__ == "__main__":
    main()

