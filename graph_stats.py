import numpy as np
import networkx as nx

# read all gmls form ./datasets/raw/ to bring stats on my graphs.
def read_graph_from_gml(file_path):
    """
    Reads a graph from a GML file and returns the graph object.
    
    Args:
        file_path (str): Path to the GML file.
    
    Returns:
        networkx.Graph: The graph object.
    """
    try:
        graph = nx.read_gml(file_path)
        return graph
    except Exception as e:
        print(f"Error reading GML file: {e}")
        return None

def calculate_graph_stats(graph):
    """
    Calculates various statistics for a given graph.
    
    Args:
        graph (networkx.Graph): The graph object.
    
    Returns:
        dict: A dictionary containing various graph statistics.
    """
    stats = {}
    stats['number_of_nodes'] = graph.number_of_nodes()
    stats['number_of_edges'] = graph.number_of_edges()
    stats['density'] = nx.density(graph)
    stats['average_clustering'] = nx.average_clustering(graph)
    stats['degree_centrality'] = nx.degree_centrality(graph)
    stats['betweenness_centrality'] = nx.betweenness_centrality(graph)
    
    return stats

def main():

    # Example usage
    file_path = './datasets/raw/your_graph.gml'  # Replace with your GML file path
    graph = read_graph_from_gml(file_path)
    
    if graph is not None:
        stats = calculate_graph_stats(graph)
        print("Graph Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("Failed to read the graph.")
