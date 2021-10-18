# script for an easy DeepWalk over karate
import random
from gensim.models import Word2Vec
import os
import pandas as pd
import networkx as nx

# HELPER FUNCTION
def random_walk(graph, walk_length, walk_number):
    r""" function to create random walks for a given graph

    Parameters
    ----------
    graph: nx.Graph, input graph
    walk_length: int, length of a single walk, how many vertices to explore
    walk_number: int, how many random walks

    Return
    -------
    random_walk: list, list of random walks
    """
    random_walks = []
    for node in graph.nodes():
        for w_numb in range(walk_number):
            # create all the walks
            path = [node]
            for w_leng in range(walk_length -1):
                # find neighbours from starting point
                neighs = [neigh_node for neigh_node in graph.neighbors(path[-1])]
                if len(neighs) > 0:
                    # add the random shuffling at this stagi
                    path = path + random.sample(neighs, 1)
            # convert to sequence of string
            path = [str(w) for w in path]
            # append
            random_walks.append(path)

    return random_walks


# MAIN FUNCTION
save_model = 1
# read facebook dataset
edges_path = 'data/facebook_large/musae_facebook_edges.csv'
targets_path = 'data/facebook_large/musae_facebook_target.csv'
features_path = 'data/facebook_large/musae_facebook_features.json'

#Read in edges
edges = pd.read_csv(edges_path)
print(f"Edges {edges.shape}")
#Read in targets
targets = pd.read_csv(targets_path)
targets.index = targets.id
print(f"Targets {targets.shape}")
graph = nx.convert_matrix.from_pandas_edgelist(edges, "id_1", "id_2")
# run the random walk across the graph
walks = random_walk(graph, 80, 10)
print(len(walks))
# and run the deepwalk skipgram
# we want to map the input social relation to a 100 dimension domain
representation_size = 100
# let's say the skipgram to look around 5 words from a given one
window_size = 5 # 40/80
# increase the cpus!
workers = 4 # cpus
model = Word2Vec(walks, # sentences
                 vector_size=representation_size, # latent dimension size
                 window=window_size, # distance between current and predicted word within the sentence
                 min_count=0, # 0 consider all the words, otherwise ignore the ones with frequences < min_count
                 sg=1, # skip gram training option
                 hs=1, # 1 hierarchical softmax used for training
                 epochs=2, # just run 2 epochs
                 workers=workers,
                 seed=42 # set the random seed for reproducibility
                 )

if save_model:
    # this is to save embeddings
    if not os.path.exists("output"):
        os.makedirs("output")

    model.wv.save_word2vec_format("output/facebook.embeddings")
