import itertools
import os

import argparse
import habitat_sim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import magnum as mn
from hb import create_viewer


matplotlib.use('TkAgg')



def manual_region_lookup(point, semantic_scene, margin = 0.0 ,y_margin=0.25):
    for idx, region in enumerate(semantic_scene.regions):
        aabb = region.aabb
        if (aabb.min.x - margin <= point.x <= aabb.max.x + margin and
                aabb.min.z - margin <= point.z <= aabb.max.z + margin and
                aabb.min.y - y_margin <= point.y <= aabb.max.y + y_margin):
            return idx, region.category.name()
    return 999999, "unknown"

def adaptive_poisson_sample_with_clearance(sim, pathfinder, num_nodes, alpha = 0.3, r_min = 0.05, r_max = 2.5,max_attempts_per_point = 50):
    samples = []
    radii = []
    attempts = 0
    max_total_attempts = num_nodes * max_attempts_per_point
    while len(samples) < num_nodes and attempts < max_total_attempts:
        p = pathfinder.get_random_navigable_point()
        dict_result = calculate_mean_clearance(sim,p)
        mean_clearance = dict_result['mean_distance']

        # Use your ray-based clearance to control local radius
        r_p = np.clip(alpha * mean_clearance, r_min, r_max)

        # Check that p is far enough from all existing points
        too_close = False
        for s, r_s in zip(samples, radii):
            path = habitat_sim.ShortestPath()
            path.requested_start = s
            path.requested_end = p
            if pathfinder.find_path(path):
                distance = path.geodesic_distance
                if distance < max(r_p, r_s):
                    too_close = True
                    break

        if not too_close:
            samples.append(p)
            radii.append(r_p)

        attempts += 1

    return samples

def calculate_mean_clearance(sim, node, max_distance = 20):
    directions = [
        np.array([1, 0, 0]),  # east
        np.array([1, 0, 1]),  # northeast
        np.array([0, 0, 1]),  # north
        np.array([-1, 0, 1]),  # northwest
        np.array([-1, 0, 0]),  # west
        np.array([-1, 0, -1]),  # southwest
        np.array([0, 0, -1]),  # south
        np.array([1, 0, -1]),  # southeast
    ]
    directions = [d / np.linalg.norm(d) for d in directions]
    distances = []
    for d in directions:
        # Using a placeholder cast_ray method from pathfinder.
        # Replace this with the actual ray-casting method from your environment if needed.
        ray_input = habitat_sim.geo.Ray(node, d)
        ray_result = sim.cast_ray(ray_input)

        if ray_result.has_hits():
            distance = ray_result.hits[0].ray_distance
        else:
            distance = max_distance
        distances.append(distance)
    # Remove the minimum and maximum distances.
    if len(distances) > 2:
        valid_distances = np.delete(distances, [np.argmin(distances), np.argmax(distances)])
    else:
        valid_distances = distances
    mean_distance = np.mean(valid_distances)
    result = {'mean_distance': mean_distance,
                'distances': distances}
    return result


def poisson_disk_sample(pathfinder, radius=1.0, max_samples=200):
    samples = []
    attempts = 0
    max_attempts = max_samples * 10
    while len(samples) < max_samples and attempts < max_attempts:
        pt = pathfinder.get_random_navigable_point()
        samples.append(pt)
        attempts += 1
    return samples

def estimate_num_nodes(pathfinder, avg_spacing = 0.25):
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


def point_on_path(pt, path_points, tolerance):
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

def create_graph_based_scene(scene_path, config_path,distance_threshold = 3.50 ,block_tolerance = 0.25, save_path = None):
    # get the id of the scene
    scene_id = os.path.basename(os.path.dirname(scene_path))

    # Basic simulator config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True
    # configuration file
    sim_cfg.scene_dataset_config_file = config_path

    # Create the simulator
    agent_cfg = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    # semantic scene label for the graph
    semantic_scene = sim.semantic_scene

    # get the pathfinder
    pathfinder = sim.pathfinder

    # define the number of the nodes N
    N = estimate_num_nodes(pathfinder)
    # poisson
    # nodes = adaptive_poisson_sample(pathfinder, N)
    nodes= adaptive_poisson_sample_with_clearance(sim,pathfinder,N,0.8)


    # create the graph
    graph = nx.Graph()
    # label the graph a scene label
    graph.graph["scene"] = scene_id

    for i, point in enumerate(nodes):
        region_id, region_name = manual_region_lookup(point, semantic_scene)
        graph.add_node(
            i,
            position=point,
            region_id=region_id,
            region_name=region_name,
        )

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


    # ---------------------SHOW GRAPH---------------------------
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

    # print the node positions
    for node_id, data in graph.nodes(data=True):
        print(f"Node {node_id} → position: {data['position']}")

    # ------------------- save -------------------------
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



def add_attributes_to_graph(graph,out_path):
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
    # need 4 different angle rendering view image
    view_angles = {
        "front": 0,
        "right": -90,
        "back": 180,
        "left": 90
    }
    for node_id, data in graph.nodes(data=True):
        position = data["position"]
        viewer.transit_to_goal(position)
        image_paths = {}
        # Save image for each direction
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