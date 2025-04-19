import os
import glob
import igraph as ig
import leidenalg
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm

# Parameters
test_id     = "17DRP5sb8fy"
test_file   = "17DRP5sb8fy_level0_navgraph.gml"
RAW_DIR     = 'datasets/raw'
RESULT_DIR  = './results'
RESOLUTIONS = [0.1, 0.5, 1.0, 2.0, 5.0]
SEED        = 12345

# Ensure result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Locate graph file (.graphml or .gml)
candidate = glob.glob(os.path.join(RAW_DIR, test_file))
if not candidate:
    raise FileNotFoundError(f"No graph file found: {test_file} in {RAW_DIR}")
graph_path = candidate[0]

# Load NetworkX graph
try:
    G_nx = nx.read_graphml(graph_path)
except Exception:
    G_nx = nx.read_gml(graph_path)

# Extract 2D positions from 'position' attribute
pos = {}
for n, d in G_nx.nodes(data=True):
    coord_str = d.get('position', '0,0,0')
    x, y, _ = list(map(float, coord_str.split(',')))
    pos[n] = (x, -d.get('position', '0,0,0').split(',')[2]) if False else (x, -float(coord_str.split(',')[2]))

# Plotting function: semantic regions + community IDs
def plot_semantic_with_communities(G, pos2d, out_path):
    # Semantic labels → shapes & colors
    labels = nx.get_node_attributes(G, 'region_name')
    unique = list(set(labels.values()))
    shapes = itertools.cycle(['o','s','^','D','v','h','*','p','X','<','>'])
    cmap = cm.get_cmap('tab20', len(unique))
    label2shape = dict(zip(unique, shapes))
    label2color = dict(zip(unique, cmap(range(len(unique)))))

    fig, ax = plt.subplots(figsize=(12,10))
    # Draw edges
    nx.draw_networkx_edges(G, pos2d, edge_color='gray', ax=ax)
    # Draw nodes by semantic region
    for lbl in unique:
        nodes = [n for n,v in labels.items() if v == lbl]
        nx.draw_networkx_nodes(
            G, pos2d,
            nodelist=nodes,
            node_shape=label2shape[lbl],
            node_color=[label2color[lbl]],
            node_size=80,
            label=lbl,
            ax=ax
        )
    # Annotate community ID
    comm = nx.get_node_attributes(G, 'community')
    for n, c in comm.items():
        x, y = pos2d[n]
        ax.text(
            x, y,
            str(c),
            fontsize=20,
            ha='center', va='center',
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.1)
        )
    ax.legend(title='Region')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# Convert once to igraph
graph_ig = ig.Graph.from_networkx(G_nx)

# Loop over resolutions
for res in RESOLUTIONS:
    # Leiden partitioning
    partition = leidenalg.find_partition(
        graph_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=res,
        seed=SEED
    )
    membership = partition.membership
    # Assign to NetworkX graph
    for node, com in zip(G_nx.nodes(), membership):
        G_nx.nodes[node]['community'] = com

    # Plot and save
    out_file = f"{test_id}_semantic_com_res_{res}.png"
    out_path = os.path.join(RESULT_DIR, out_file)
    plot_semantic_with_communities(G_nx, pos, out_path)
    print(f"Saved {out_path}")