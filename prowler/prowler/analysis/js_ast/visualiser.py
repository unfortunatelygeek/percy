import networkx as nx
import matplotlib.pyplot as plt

def visualize_vulnerabilities(G, output_file=None):
    """Visualizes AST highlighting vulnerable nodes."""
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)

    normal_nodes = [n for n in G.nodes if not G.nodes[n]['vulnerabilities']]
    vuln_nodes = [n for n in G.nodes if G.nodes[n]['vulnerabilities']]

    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color='lightblue', node_size=1000, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=vuln_nodes, node_color='red', node_size=1200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.0, arrows=True)

    labels = {node: G.nodes[node]['label'] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.axis('off')
    plt.title("Code Vulnerability Analysis")

    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()
