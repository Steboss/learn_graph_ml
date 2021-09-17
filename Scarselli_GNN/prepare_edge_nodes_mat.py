# Prepare E and N matrices
import scipy.sparse as sp
import numpy as np
import random

def preprocess():
    r""" in this function we are preprocessing the input info for GNN
    Edges are written in data/edges.txt file
    Each node has its label, form 0 to 3. The idea in the GNN code is:
    - read the edges
    - create a set of features
    - read the labels
    - hot-encode the labels for each node, so node 0 may be [0, 1, 0, 0]
    - Create matrix E = edges | index of the graph (0, namely everyone in the same graph)
    - Create matrix N = nodes' features | index of the graph for the node (0)
    - create a mask for training nodes

    Parameters
    ----------

    Return
    ------
    E: np.array, edges array, last column is graph index (0)
    N: np.array, nodes array, last column is graph index (0)
    labels: np.array, one-hot encoding array with labels [0-3]
    mask_train: np.array, training nodes
    mask_test: np.array, test nodes
    """

    edges = np.loadtxt("data/edges.txt", dtype=np.int32) -1 # -1 as it's 0 index based
    edges = edges[np.lexsort((edges[:,1], edges[:,0]))] # reorder the edges
    # create a one-hot encoding sparse matrix for features
    features = sp.eye(np.max(edges+1), dtype=np.float32).tocsr()
    # extract labels for each node
    idx_labels = np.loadtxt("data/labels.txt", dtype=np.int32)
    idx_labels = idx_labels[idx_labels[:,0].argsort()]
    labels = np.eye(max(idx_labels[:,1])+1, dtype=np.int32)[idx_labels[:,1]] # one-hot encoding of labels

    # create edges matrix + graph id
    E = np.concatenate((edges, np.zeros((len(edges), 1), dtype=np.int32)), axis=1)
    # create the nodes' feature matrix + graph id
    N = np.concatenate((features.toarray(), np.zeros((features.shape[0], 1), dtype=np.int32)), axis=1)
    # create mask train and mask test
    mask_train = np.zeros(shape=(34,), dtype=np.float32)
    idx_classes = np.argmax(labels, axis=1)

    id_0, id_4, id_5, id_12 = random.choices(np.argwhere(idx_classes == 0), k=4)
    id_1, id_6, id_7, id_13 = random.choices(np.argwhere(idx_classes == 1), k=4)
    id_2, id_8, id_9, id_14 = random.choices(np.argwhere(idx_classes == 2), k=4)
    id_3, id_10, id_11, id_15 = random.choices(np.argwhere(idx_classes == 3), k=4)

    mask_train[id_0] = 1.  # class 1
    mask_train[id_1] = 1.  # class 2
    mask_train[id_2] = 1.  # class 0
    mask_train[id_3] = 1.  # class 3
    mask_test = 1. - mask_train

    return E, N, labels, mask_train, mask_test