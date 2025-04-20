#!/usr/bin/env python3
import os
import glob
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory paths
raw_dir = 'datasets/raw'
result_dir = './results'
os.makedirs(result_dir, exist_ok=True)

# Collect all graph file paths (support both .gml and .graphml)
file_paths = glob.glob(os.path.join(raw_dir, '*.gml')) + glob.glob(os.path.join(raw_dir, '*.graphml'))

# Initialize metric lists
metrics = {
    'num_nodes': [],
    'num_edges': [],
    'density': [],
    'avg_clustering': [],
    'transitivity': []
}

# Process each graph
for path in file_paths:
    try:
        G = nx.read_graphml(path)
    except:
        G = nx.read_gml(path)
    metrics['num_nodes'].append(G.number_of_nodes())
    metrics['num_edges'].append(G.number_of_edges())
    metrics['density'].append(nx.density(G))
    metrics['avg_clustering'].append(nx.average_clustering(G))
    metrics['transitivity'].append(nx.transitivity(G))

# Check if any graphs were found
if not metrics['num_nodes']:
    print(f"No graph files found in {raw_dir}")
else:
    # Compute summary statistics
    summary = []
    for name, vals in metrics.items():
        arr = np.array(vals)
        summary.append({
            'metric': name,
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
            'std': float(arr.std())
        })
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    # Plot histograms for each metric
    for name, vals in metrics.items():
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=20, edgecolor='black')
        plt.xlabel(name.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.title(f"Distribution of {name.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{name}_hist.png"))
        plt.close()

    print(f"Histograms saved to {result_dir}")