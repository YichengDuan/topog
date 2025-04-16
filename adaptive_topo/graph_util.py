import numpy as np
import habitat_sim


def estimate_num_nodes(pathfinder, avg_spacing = 0.35):
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

def sampling_nav(sampling_way, num_nodes, pathfinder, sim):
    """
    :param sampling_way: different way to sampling the graph (0: random sample, 1: fixed radius poisson sample, 2: adaptive_poisson_sample)
    :param num_nodes: the expected number of nodes
    :param pathfinder: habitat_sim.PathFinder object
    :param sim: the habitat_sim object
    """
    if sampling_way == 0:
        # random sampling
        nodes = random_sample(pathfinder,num_nodes)
    elif sampling_way == 1:
        # fixed radius poisson sample
        nodes = fixed_poisson_disk_sample(pathfinder, num_nodes)
    elif sampling_way == 2:
        # adaptive_poisson_sample(default: the ray way)
        nodes = adaptive_poisson_sample(1,sim,pathfinder,num_nodes,alpha = 0.35)
    else:
        raise ValueError(f"Unsupported sampling_way: {sampling_way}")
    return nodes


## WAY 1: Randomly Sampling
def random_sample(pathfinder, num_nodes: int):
    samples = []
    attempts = 0
    max_attempts = num_nodes * 10  # Avoid infinite loops
    while len(samples) < num_nodes and attempts < max_attempts:
        pt = pathfinder.get_random_navigable_point()
        if not any(np.allclose(pt, s['point'], atol=1e-3) for s in samples):  # avoid near-duplicates
            sample = {'point': pt, 'radius': None}
            samples.append(sample)
        attempts += 1
    return samples

## WAY 2: Fixed Radius Poisson disk sampling
def fixed_poisson_disk_sample(pathfinder, num_nodes: int, radius=0.5, max_attempts_per_point = 30):
    samples = []
    attempts = 0
    max_attempts = num_nodes * max_attempts_per_point
    while len(samples) < num_nodes and attempts < max_attempts:
        pt = pathfinder.get_random_navigable_point()
        too_close = False
        for s in samples:
            if np.linalg.norm(np.array(pt) - np.array(s)) < radius:
                too_close = True
                break
        if not too_close:
            sample = {'point':pt, 'radius':radius}
            samples.append(sample)
        attempts += 1
    return samples

## WAY 3: Adaptive Poisson disk sampling(two different radius)
def adaptive_poisson_sample(option, sim, pathfinder, num_nodes, alpha, r_min = 0.05, r_max = 2.5,max_attempts_per_point = 30):
    """
    :param option: the optional for the radius. 0 for the way using the closed distance to obstacle, 1 for the ray way
    :param sim: habitat_sim object
    :param pathfinder: habitat_sim.PathFinder object
    :param num_nodes: number of nodes to sample
    :param alpha: scale factor for adaptive radius
    :param r_min: minimum radius
    :param r_max: maximum radius
    :param max_attempts_per_point: attempts per point before giving up
    """
    samples = []
    radii = []
    attempts = 0
    max_total_attempts = num_nodes * max_attempts_per_point
    while len(samples) < num_nodes and attempts < max_total_attempts:
        p = pathfinder.get_random_navigable_point()

        if option == 0:
            # using the closed distance to obstacle
            d =  pathfinder.distance_to_closest_obstacle(p)
            r_p = np.clip(alpha * d, r_min, r_max)
        else:
            # using ray
            dict_result = calculate_mean_clearance(sim, p)
            mean_clearance = dict_result['mean_distance']
            # Use your ray-based clearance to control local radius
            r_p = np.clip(alpha * mean_clearance, r_min, r_max)
        # Check that p is far enough from all existing points
        too_close = False
        for s, r_s in zip(samples, radii):
            path = habitat_sim.ShortestPath()
            path.requested_start = s['point']
            path.requested_end = p
            euclidean_dist = np.linalg.norm(np.array(s['point']) - np.array(p))
            if pathfinder.find_path(path):
                distance = path.geodesic_distance
                if euclidean_dist < max(r_p, r_s):
                    too_close = True
                    break
        if not too_close:
            sample = {'point':p, 'radius':r_p}
            samples.append(sample)
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


# ----------------------- EDGE ---------------------
# WAY 1 :  using geodesic and remove the edge if blocked by other nodes
def add_edge(pathfinder,graph,nodes,distance_threshold = 2.5, block_tolerance = 0.30):
    # For each pair of nodes within distance threshold, check if there's a path
    # If so, add an edge with geodesic (shortest-path) distance as the edge weight
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            path = habitat_sim.ShortestPath()
            path.requested_start = nodes[i]
            path.requested_end = nodes[j]

            if pathfinder.find_path(path):
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
    return graph


# WAY 2:
def add_edge_ray(pathfinder, graph, nodes_info, distance_threshold = 0.75, tolerance = 1.05):
    for i in range(len(nodes_info)):
        pos_i = nodes_info[i]['point']
        radius_i = nodes_info[i]['radius']
        for j in range(i + 1, len(nodes_info)):
            radius_j = nodes_info[j]['radius']
            pos_j = nodes_info[j]['point']
            path = habitat_sim.ShortestPath()
            path.requested_start = pos_i
            path.requested_end = pos_j
            if pathfinder.find_path(path):
                euclidean_dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                geodestic_dist = path.geodesic_distance
                if  geodestic_dist <  tolerance * euclidean_dist:
                    # the path have no curve
                    if geodestic_dist < 1.25*(radius_i + radius_j):
                        graph.add_edge(i, j, weight=path.geodesic_distance)
    return graph


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


# --------------------------Semantic-------------------------------------

def manual_region_lookup(point, semantic_scene, margin = 0.0 ,y_margin=0.25):
    for idx, region in enumerate(semantic_scene.regions):
        aabb = region.aabb
        level = region.level.id
        if (aabb.min.x - margin <= point.x <= aabb.max.x + margin and
                aabb.min.z - margin <= point.z <= aabb.max.z + margin and
                aabb.min.y - y_margin <= point.y <= aabb.max.y + y_margin):
            return idx, region.category.name(),level
    return 999999, "unknown", "unknown"

#-------------------------------visualization-------------------------
