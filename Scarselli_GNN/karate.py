from prepare_edge_nodes_mat import preprocess
from prepare_GNN import from_EN_to_GNN
import input_and_output_functions as net
import GNN
from scipy.sparse import coo_matrix
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# create the input matrices
E, N, labels, mask_train, mask_test = preprocess()
# transform to graph nn
inp, arcnode, graphnode = from_EN_to_GNN(E, N)

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.001
learning_rate = 0.001
state_dim = 2
input_dim = inp.shape[1]
output_dim = labels.shape[1]
max_it = 50
num_epoch = 10000
optimizer = tf.compat.v1.train.AdamOptimizer

# initialize state and output network
net = net.Net(input_dim, state_dim, output_dim)

# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)

tensorboard = True

g = GNN.GNN(net,
            input_dim,
            output_dim,
            state_dim,
            max_it,
            optimizer,
            learning_rate,
            threshold,
            graph_based=False,
            param=param,
            config=None,
            tensorboard=tensorboard,
            mask_flag=True)


# train the model
count = 0

######

for j in range(0, num_epoch):
    _, it = g.Train(inputs=inp, ArcNode=arcnode, target=labels, step=count, mask=mask_train)

    if count % 10 == 0:
        print("Epoch ", count)
        print("Training: ", g.Validate(inp, arcnode, labels, count, mask=mask_train))
        print("Test: ", g.Validate(inp, arcnode, labels, count, mask=mask_test))

        # end = time.time()
        # print("Epoch {} at time {}".format(j, end-start))
        # start = time.time()

    count = count + 1
