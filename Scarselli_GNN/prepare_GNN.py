# here from matrices E and N we are going to make up the final graph target NN
import numpy as np
import scipy.sparse as sp
from collections import namedtuple
SparseMatrix = namedtuple("SparseMatrix", "indices values dense_shape")


def from_EN_to_GNN(E, N):
    r""" This function converts the nodes and features matrices to a Graph Neural Network structure
    From this function we want to retur:
    the input graph for the neural network
    the structure edge structure for neural network with 1 for source nodes
    the graphnode matrix, namely an identity matrix based on the graph_id - useful for graph focused cases"""
    print("Transforming matrices E and N to GNN")
    N_full = N
    N = N[:, :-1] # forget the graph_id (nodes' features) --> 34x34
    print("Matrix nodes' features N shape")
    print(N.shape)
    e = E[:, :2] # forget the graph_id (edges) --> 78x2
    print("Matrix edges shape")
    print(e.shape)
    # now take the matrix N and create an array which has the fetures of node A and node B
    # for any pair we have in e
    # for example N[0] = [1, 0, 0, blabla] N[1] = [0, 1, 0, blabla]
    # e[0] = [0,1]
    # so now create an array like:
    # array[0] = [ [1, 0, 0, blabla], [0, 1, 0, blabla] ] which is N[0] and N[1], with indices extracted from e
    # this result in an array of shape 78 (the number of edges) x 2 ( in each row we have 2 arrays which are the
    # nodes' features ) x 34 which is the size of each row (34 features)
    # to do this use take
    feat_temp = np.take(N, e, axis=0) # take the indices from e and apply to N ==> n_archs x 2 x label_dimensions
    # now combine the 2 arrays in each row together
    feat = np.reshape(feat_temp, [len(E), -1]) # reshape to have n_archs x 2*label_dim  => 78x68
    # create the input for gnn [ source node, destination node, features source, features destination]
    print("creation of input for GNN")
    inp = np.concatenate((E[:, :2], feat), axis=1)
    print(inp.shape)
    print(inp)
    # create the arcnode matrix,
    # the indices are the position within the matrix we want to have values of 1
    # np.stack(blabla) we have E[:,0] which is the source node, np.arange(len(E)) is from 0 to 78
    # this creates the indices where we want 1
    # values = 1
    # shape 34 x 78  hot encoded hwo nodes are linked each other
    print("creation of the arcnode matrix, namely how the nodes are linked")
    arcnode = SparseMatrix(indices=np.stack((E[:,0], np.arange(len(E))), axis=1),
                           values = np.ones([len(E)]).astype(np.float32),
                           dense_shape=[len(N), len(E)]
                           )
    print("shape ", len(N), len(E))
    print(arcnode)
    print("Current state")
    curr_state = np.zeros((arcnode.dense_shape[0], 2))
    print(curr_state)
    # then graphnode
    # THIS IS VALID ONLY IF YOUR GRAPH AS A WHOLE HAS A LABEL - not th ecase for karate
    num_graphs = int(max(N_full[:, -1]) + 1)
    # get all graph_ids
    g_ids = N_full[:, -1] # this is 0
    g_ids = g_ids.astype(np.int32)

    # create identity matrix get row corresponding to id of the graph
    # 1 x 34
    # np.arange(len(g_ids)) from 0 to 33
    # so indices are [0,0], [0,1], [0,2] ... [0,33]
    graphnode = SparseMatrix(indices=np.stack((g_ids, np.arange(len(g_ids))), axis=1),
                             values=np.ones([len(g_ids)]).astype(np.float32),
                             dense_shape=[num_graphs, len(N)])


    return inp, arcnode, graphnode