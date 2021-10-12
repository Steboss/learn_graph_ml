# script for an easy DeepWalk over karate
import random
from graph import Graph
import os
from six.moves import range, zip, zip_longest
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# HELPER FUNCTION
def visualize_karate(inputfile, outputname):
    r""" Given the inputfile return the karate graph
    This is useful to get some of the random walks as well
    Parameters
    ---------
    inputfile: str, path ot the inputfile
    outputname: str, e.g. "karate.png"

    """
    G = nx.read_adjlist(inputfile)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True)
    plt.tight_layout()
    plt.savefig(outputname, dpi=300, bbox_inches='tight')


def build_deepwalk_corpus(G, num_paths, path_length, rand=random.Random(0)):
    r""" Build teh corpus of nodes for deep walk

    Parameters
    ----------
    G: input graph (dictionary)
    num_paths: int, number of random paths
    path_length: int, length of random walks - how many nodes to span?
    rand: float, random seed
    """
    print("Build deepwalk corpus")
    walks = []

    nodes = list(G.nodes())
    print("Total number of nodes")
    print(len(nodes))
    print("Total number of random paths")
    print(num_paths)

    count_paths = 0

    for cnt in range(num_paths):
        #print("Shuffling nodes and cycling through nodes")
        rand.shuffle(nodes)  # this command shuffle the list in place creating a list rand
        count_paths+=1

        for node in nodes:
            #print(f"start node {node}")
            # here perform the random walk
            walks.append(G.random_walk(path_length, rand=rand, start=node))
        if count_paths%10==0:
            print(f"Executed {count_paths} paths")
    # return all the combinations of walks
    return walks


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G


def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist


def load_adjacencylist(file_, chunksize=10000):

    adjlist = []

    total = 0
    with open(file_) as f:
        for idx, adj_chunk in enumerate(map(parse_adjacencylist_unchecked, grouper(int(chunksize), f))):
            adjlist.extend(adj_chunk)
            total += len(adj_chunk)

    G = from_adjlist_unchecked(adjlist)

    return G
#####################################
### WORD2VEC functions ##############
def generate_training_data(random_walks, window_size):
    r""" This function generate X and Y data for the word2vec
    Parameters
    ----------
    random_walks: list, list of all the random walks
    window_size: int, dimension of the window for W2V

    Return
    ------
    """
    # TODO: add a visual here to show the paris
    X = []
    y = []
    # for every random walk we have to generate the pair X, y
    for random_walk in random_walks:
        N = len(random_walk)
        #print(random_walk)
        for i in range(N):
            idxs_left = np.arange(max(0, i-window_size), i).tolist()
            idxs_right = np.arange(i+1, min(N, i+window_size+1)).tolist()

            for j in idxs_left:
                X.append(random_walk[i])
                y.append(random_walk[j])
            for j in idxs_right:
                X.append(random_walk[i])
                y.append(random_walk[j])
            #print(X)
            #print(y)

    # convert to array
    X = np.array(X).astype('int')
    X = np.expand_dims(X, axis=0)
    y = np.array(y).astype('int')
    y = np.expand_dims(y, axis=0)

    return X, y

# NNET FOR EMBEDDINGS
def word_embedding_init(vocab_size, representation_size):
    r""" Initialize the word embeddings
    Word embeddings shape is the number of nodes * the latent dimension

    Parameters
    ----------
    vocab_size: int, size of the graph's nodes
    representation_size: int, latent dimension

    Return
    -------
    np.random.randn matrix of size vocab_size * representation_size
    """
    print("initializing rnadom embedding matrix")
    return np.random.randn(vocab_size, representation_size)*0.01


def layers_init(inp_size, out_size):
    r""" initialize dense layers weights, randomly

    Parameters
    ----------
    inp_size: int, size of the input to dense layer
    out_size: int, siez of the output

    Return
    ------
    np.random.randn random weight matrix of size out_size*inp_size
    """
    print("Initializing random weights for dense layer")
    return np.random.randn(out_size, inp_size) * 0.01


def param_init(vocab_size, representation_size):
    r""" This function initialize the word embedding matrix and the
    neural network layers weight

    Parameters
    ----------
    vocab_size: int, size of the graph's nodes
    representation_size: int, latent dimension
    """
    word_embds = word_embedding_init(vocab_size, representation_size)
    nnet_weight = layers_init(representation_size, vocab_size)

    parameters = {}
    parameters['word_embedding'] = word_embds
    parameters['nnet_weight'] = nnet_weight
    return parameters


### FORWARD NNET
def node_to_embedding(nodes, parameters):
    r""" This function takes as input the input layer nodes
    the nodes are passed to the random initialized embeddign matrix
    from there we can retrieve their latent representation value """
    node_embedding = parameters['word_embedding']
    node_vector = node_embedding[nodes.flatten(), :].T

    return node_vector # this is an array with representation_size values for each input node


def linear_dense(node_vector, parameters):
    r""" Implement a linear dense layer, which is a dot product between
    the input nodes and the layer weights

    Parameters
    ---------
    node_vector: np.array, input embedding representation of the given input nodes
    parameters: dict, current parameter weight values

    Return
    -------
    weights: np.array, weights of the dnese layer
    output: np.array, output value
    """
    weights = parameters['nnet_weight']
    # compute output
    output = np.dot(weights, node_vector)
    return weights, output


def softmax(nnet_output):
    r""" Retrieve the probability distribution of the values, given the output from the
    dense layers
    Parameters
    -----------
    nnet_output: np.array, output from dense layer, linear_dense

    Return
    ------
    softmax_out: np.array, output from softmax layers"""
    softmax_out = np.divide(np.exp(nnet_output), np.sum(np.exp(nnet_output), axis=0, keepdims=True) + 0.001)

    return softmax_out


def forward_propagation(input_nodes, parameters):
    r""" This is the main forward propagation part of the embedding neural network

    Parameters
    ----------
    input_nodes: list, input nodes
    parameters: dictionary, created through  param_init
    """
    print("retrieving embeddings")
    node_vector = node_to_embedding(input_nodes, parameters)
    print(node_vector)
    print("Computing weights & returning output from dot product with embeddings")
    weights, nnet_output = linear_dense(node_vector, parameters)
    print("Nnet output")
    print(nnet_output)
    print("softmax layer")
    softmax_out = softmax(nnet_output)
    print(softmax_out)

    updated_params = {}
    updated_params['inp_nodes'] = input_nodes
    updated_params['node_vector'] = node_vector
    updated_params['weights'] = weights
    updated_params['softmax_out'] = softmax_out

    return softmax_out, updated_params


### BACKWARD PROCESS
def dense_backward(grad_softmax, updated_params):
    r""" compute gradients in the dense layer
    In the dense layer we need to compute the gradient wrt to weights
    and wrt to word embedding

    Parameters
    ----------
    grad_softmax: float, gradient value from softmax layer
    updated_params: dict, currenct updated parameters

    Return
    ------
    grad_weight: gradient wrt to weights
    grad_embs: gradient wrt to embeddings
    """

    weights = updated_params['weights']
    node_vector = updated_params['node_vector']
    m = node_vector.shape[1]

    grad_weight = (1 / m) * np.dot(grad_softmax, node_vector.T)
    grad_embs = np.dot(weights.T, grad_softmax)


    return grad_weight, grad_embs


def softmax_backward(y, softmax_out):
    r""" Compute the gradient at softmax level
    Parameters
    ----------
    y: np.array: input target
    softmax_out: np.array, softmax values"""
    grad = softmax_out - y

    return grad


def backward_propagation(y, softmax_out, updated_params):
    r""" Main function to compute the gradient"""

    # grad of loss function w.r.t softmax
    print("Softmax gradient")
    grad_softmax = softmax_backward(y, softmax_out)
    print(grad_softmax)
    # grad of dense layers -->
    print("Dense layer gradients")
    grad_weights, grad_embs = dense_backward(grad_softmax, updated_params)
    print(f"Grad w.r.t weights {grad_weights}")
    print(f"Grad w.r.t embeddings {grad_embs}")
    gradients = dict()
    gradients['grad_softmax'] = grad_softmax
    gradients['grad_weights'] = grad_weights
    gradients['grad_embs'] = grad_embs

    return gradients


def update_parameters(parameters, updated_params, gradients, learning_rate):
    r""" Backward main funciton, update all the network parameters"""
    vocab_size, representation_size = parameters['word_embedding'].shape
    input_nodes = updated_params['inp_nodes']
    node_embedding = parameters['word_embedding']
    gradient_embedding = gradients['grad_embs']

    node_embedding[input_nodes.flatten(), :] -= gradient_embedding.T * learning_rate

    parameters['nnet_weight'] -= learning_rate * gradients['grad_weights']

### LOSS
def cross_entropy(softmax_out, y):
    r""" define the loss function, which is the cross entropy
    Parmeters
    ---------
    softmax_out: float, output value from softmax layer
    y: np.array, target values

    Return
    ------
    cost: float, loss cost"""
    m = softmax_out.shape[1]
    cost = -(1 / m) * np.sum(np.sum(y* np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
    print(cost)
    return cost

# MAIN FUNCTION

save_model = 1
inputpath = "data/karate.adjlist"
# read the edge list
G = load_adjacencylist(inputpath)
print("Number of nodes: {}".format(len(G.nodes())))
# representation
if os.path.exists("karate_expl"):
    pass
else:
    os.makedirs("karate_expl")

visualize_karate(inputpath, "karate_expl/karate.png")
# define the number of walks
number_walks = 5
num_walks = len(G.nodes()) * number_walks
print("Number of walks: {}".format(num_walks))
# check the data size
walk_length = 5
data_size = num_walks * walk_length
print(f"Data size (walks*length): {data_size/1000000} MB")
# depending on the data size different solutions can be used
walks = build_deepwalk_corpus(G,
                              num_paths = number_walks,
                              path_length = walk_length,
                              rand = random.Random(42)
                              )

print(f"Total number of walks {len(walks)}, first 10 elements {walks[0:10]}")

# a simple setch about how the Word2Vec works
representation_size = 2  # latent dimension
window_size = 5  # window size before and after a vertex
X, y = generate_training_data(walks, window_size)
vocab_size = len(G.nodes())+1
print(X.shape, y.shape)
# one-hot encoding for Y
Y_one_hot = np.zeros((vocab_size, y.shape[1]))
Y_one_hot[y.flatten(), np.arange(y.shape[1])] = 1
print(Y_one_hot)
# Word2Vec Skipgram
epochs=2
m = X.shape[1]
batch_size=4096
parameters = param_init(vocab_size, representation_size)
learning_rate = 0.001
costs = []
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    epoch_cost = 0
    # define the batch indexes to investigate
    batch_idxs = np.arange(0, m, batch_size)
    #np.random.shuffle(batch_idxs)
    for i in batch_idxs:
        X_batch = X[:, i:i + batch_size]
        Y_batch = y[:, i:i + batch_size]
        #
        print(f"Start Word2Vec for batch {i}")
        softmax_out, updated_params = forward_propagation(X_batch, parameters)
        print("Computing gradients")
        gradients = backward_propagation(Y_batch, softmax_out, updated_params)
        print("Update parameters")
        update_parameters(parameters, updated_params, gradients, learning_rate)
        print("Computing cost")
        cost = cross_entropy(softmax_out, Y_batch)
        epoch_cost += np.squeeze(cost)

    costs.append(epoch_cost)
