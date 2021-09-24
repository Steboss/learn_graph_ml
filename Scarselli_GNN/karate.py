from prepare_edge_nodes_mat import preprocess
from prepare_GNN import from_EN_to_GNN
import input_and_output_functions as net
import GNN
from create_and_visualize import draw, build_karate_club_graph
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from datetime import datetime
import matplotlib.pyplot as plt
import os
import imageio

# create the initial graph
G = build_karate_club_graph()
nx_G = G.to_networkx().to_undirected()
# create the input matrices
E, N, labels, mask_train, mask_test = preprocess()
# transform to graph nn
inp, arcnode, graphnode = from_EN_to_GNN(E, N)

# set parameters
threshold = 0.0001
learning_rate = 0.001
state_dim = 2
input_dim = inp.shape[1]
output_dim = labels.shape[1]
max_it = 20 # max number of iterations
num_epoch = 200
optimizer = tf.compat.v1.train.AdamOptimizer
tensorboard = True # False if you don't want to use tensorboard
savegif = 0 # 0 if you do not want to save gif, 1 otherwise
# initialize state and output network
net = net.Net(input_dim, state_dim, output_dim)
# define the name of the experiment, so it can be visualized in tensorboard
today = datetime.today().strftime("%Y-%m-%d")
param = "st d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate) + today
print("tensorboard: ", param)
print("Initialize the graph neural network")
g = GNN.GNN(net,
            input_dim,
            output_dim,
            state_dim,
            param=param,
            tensorboard=tensorboard,
            mask_flag=True)

# train the model
count = 0
# initialise an empty list to store nodes positions during training
probs = []
fig = plt.figure(dpi=150)
ax = fig.subplots()

# create a imgs folder for gif reation
if os.path.exists("imgs"):
    pass
else:
    os.makedirs("imgs")

# start the training
for j in range(0, num_epoch):
    # _ is the loss
    _, loop_val = g.Train(inputs=inp,
                    ArcNode=arcnode,
                    target=labels,
                    step=count,
                    mask=mask_train)
    # append all the probs during training
    if savegif:
        probs.append(loop_val[2])
        ax = draw(j, probs, ax, nx_G)
        img_name = f"imgs/{j}.png"
        fig.savefig(img_name, dpi=300, bbox_inches='tight',)
    # print out a statement every 100 epochs
    if count % 100 == 0:
        print("Epoch ", count)
        print(loop_val)
        print("Training: ", g.Validate(inp, arcnode, labels, count, mask=mask_train))

    count = count + 1

# final save gif
if savegif:
    # Build GIF
    with imageio.get_writer('final_gif.gif', mode='I') as writer:
        for filename in range(0, num_epoch):
            file_path = f"imgs/{filename}.png"
            image = imageio.imread(file_path)
            writer.append_data(image)