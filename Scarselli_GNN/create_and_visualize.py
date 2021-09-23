# Use DGL to create the graph
# DGL is a nice package with lovely documentation for deep learning on graphs , built on top of PyTorch and MXNet
import dgl
# networkx helps in visualizing the graph
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# FUNCTIONS
def draw(i, probs, ax, nx_G):
    r""" Function to draw the nodes' state after each epoch

    Parameters
    ----------
    i: int, number of epoch
    probs: np.array, array of nodes' state after each epoch
    ax: matplotlib.pyplot.subplots(), axes for drawing
    nx_G: networkx, current karate graph, nodes and edges

    Return
    -------
    ax: update graph
    """
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    # 34 number of nodes
    for v in range(34):
        pos[v] = probs[i][v]#.numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)
    return ax

def build_karate_club_graph():
    r""" This function builds up the network in the karate club
    Parameters
    ----------

    Return
    -------
    g: dgl.DGLGraph, graph from given vertices and edges
    """

    # initialize an empty graph
    g = dgl.DGLGraph()
    # we have 34 nodes in the club, labelled 0~33
    g.add_nodes(34)
    # the 34 nodes are linked with 78 edges, social relationships outside the club
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0),
                 (7, 1), (7, 2), (7, 3), (8, 0), (8, 2), (9, 2),
                 (10, 0), (10, 4), (10, 5), (11, 0), (12, 0), (12, 3),
                 (13, 0), (13, 1), (13, 2), (13, 3), (16, 5), (16, 6),
                 (17, 0), (17, 1), (19, 0), (19, 1), (21, 0), (21, 1),
                 (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)
                 ]
    # now create nodes and edges. zip the list above, so the first index will be source nodes and the
    # second index will be the destination node
    src, dst = tuple(zip(*edge_list))
    # now that we have define the source nodes and the destination ones we can insert the edge info in the graph
    g.add_edges(src, dst)
    # in this case the edges can be bi-directional, so going from src to dst and from dst to src
    g.add_edges(dst, src)

    return g


def visualize_karate_club():
    r""" This function calls build_karate_club_graph and displays the final network
    Parameters
    ----------

    Return
    -------
    """

    G = build_karate_club_graph()
    nx_G = G.to_networkx().to_undirected()
    # use the Kamade-Kawaii layout to display the graph
    pos = nx.kamada_kawai_layout(nx_G)
    # create the figure
    fig, ax = plt.subplots()
    ax.set_title("Karate Club", fontsize=16)
    nx.draw(nx_G, pos, with_labels=True)
    # save
    plt.tight_layout()
    plt.savefig("tester.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# LOCAL
if __name__ == '__main__':
    visualize_karate_club()