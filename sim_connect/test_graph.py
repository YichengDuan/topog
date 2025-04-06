import os
import time
import numpy as np
import networkx as nx
from hb import create_viewer, command_queue  # your viewer init
import threading

def walk_graph_path(viewer, graph_path, start_id, goal_id, sleep_time=0.5):
    """
    Given the node id from the graph,calculate the shortest path in the graph and walk to the goal through the path

    Args:
        viewer: An initialized viewer.
        graph_path: Path to the graph in GraphML format.
        start_id: ID of the starting node.
        goal_id: ID of the goal node.
        sleep_time: Time to sleep between steps (in seconds).
    """
    # load the graph
    graph = nx.read_graphml(graph_path)
    if start_id not in graph.nodes or goal_id not in graph.nodes:
        print(f"Invalid start or goal node. Available nodes: {list(graph.nodes)}")
        return
    # Compute path
    try:
        path = nx.shortest_path(graph, source=start_id, target=goal_id, weight='weight')
    except nx.NetworkXNoPath:
        print("No path found in the graph.")
        return
    print(f"Path from node {start_id} to {goal_id}: {path}")

    # Get start position from graph
    start_pos_str = graph.nodes[start_id]['position']
    start_pos = np.array([float(v) for v in start_pos_str.split(",")])
    # Teleport agent to start node
    viewer.transit_to_goal(start_pos)
    time.sleep(3)

    # Walk through each node in the path
    for step, node_id in enumerate(path):
        pos_str = graph.nodes[node_id]['position']
        pos = np.array([float(v) for v in pos_str.split(",")])
        print(f"Step {step}: Node {node_id} → {pos}")
        cmd_thread = threading.Thread(target=viewer.move_to_goal, args=(pos,), daemon=True)
        cmd_thread.start()
        cmd_thread.join()

        time.sleep(sleep_time)

    print("Reached the goal!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",default="../data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb", type=str)
    parser.add_argument("--graph",default="../data/out/graph/17DRP5sb8fy_navgraph.gml", type=str)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--goal", type=str, required=True)
    args = parser.parse_args()

    viewer = create_viewer(args.scene)
    threading.Thread(
        target=walk_graph_path,
        args=(viewer, args.graph, args.start, args.goal),
        daemon=True
    ).start()
    viewer.exec()
