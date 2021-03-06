# here we are defining the function fw and gw as neural networks
import tensorflow as tf


class Net:
    r""" In this class we are defining the state and output functions (fw and gw respectively)"""

    def __init__(self, input_dim, state_dim, output_dim):
        r"""Constructor

        Parameters
        ----------
        input_dim: input dimension of the graph, in our case is 70
        state_dim: this is 2 in this case, as we are predicting 1/0 labels
        output_dim: the output must be the same as the labels, in our case 4
        """
        print("Initialize the net functions")
        # define an epsilon variable for the iteration procedure
        self.EPSILON = 0.00000001
        # input & output
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_input = self.input_dim - 1 + state_dim  # removing the id_ dimension


        # you have to play a bit with this number to find the right value
        self.state_l1 = 3
        self.state_l2 = self.state_dim

        self.output_l1 = 3
        self.output_l2 = self.output_dim


    def netSt(self, inp):
        r""" Define the state function fw
        The inp matrix, features, is processed initially. state_l1 is fixed to 5,
        state_l2 is equal to state dimension, which is 2

        Parameters
        ----------
        inp: self.input_dim, dimension edges x (edges+source_features+destination_features)

        Return
        ------
        layer2: tf.compat.v1.layers.dense output of second layer
        """
        print("Define the nn for fw")
        with tf.compat.v1.variable_scope('State_net'):
            # tanh activation function, 2 dense layers
            layer1 = tf.compat.v1.layers.dense(inp, self.state_l1, activation=tf.nn.tanh)
            layer2 = tf.compat.v1.layers.dense(layer1, self.state_l2, activation=tf.nn.tanh)

            return layer2

    def netOut(self, inp):
        r""" Define the output function gw
        The input dimensions are 2, as they come from netSt output. output_l1 is set to 5,
        output_l2 is the number of labels which is 2

        Parameters
        ----------
        inp: self.input_dim

        Return
        -------
        layer2: tf.compat.v1.layers.dense: output_l2 dimension array with prediction for each node
        """
        print("Define the nn for gw")
        with tf.compat.v1.variable_scope('Output_net'):
            layer1 = tf.compat.v1.layers.dense(inp, self.output_l1, activation=tf.nn.tanh)
            layer2 = tf.compat.v1.layers.dense(layer1, self.output_l2, activation=tf.nn.softmax)

        return layer2

    def Loss(self, output, target, output_weight=None, mask=None):
        r""" loss function for the neural network structure

        Parameters
        ----------
        output: current prediction from the gw function
        target: real label for the nodes
        output_weight: None, used only if we have weights
        mask: None, supervised nodes
        """
        print("Initialize the loss function")
        # method to define the loss function
        # which is the max between the epsilon threshold and the current output from state
        output = tf.maximum(output, self.EPSILON, name="Avoiding_explosions")  # to avoid explosions
        # sum to the current state target*log(output)
        xent = -tf.reduce_sum(target * tf.math.log(output), 1)
        # apply only to the current mask
        mask = tf.cast(mask, dtype=tf.float32)
        # normalize the mask values
        mask /= tf.reduce_mean(mask)
        # compute the x state
        xent *= mask
        # loss
        lo = tf.reduce_mean(xent)

        return lo

    def Metric(self, target, output, output_weight=None, mask=None):
        r""" Here compute the accuracy of predictions for current state

        Parameters
        ----------
        target: real label for the nodes
        output: current prediction from the gw function
        output_weight: None, used only if we have weights
        mask: None, supervised nodes
        """
        # method to define the evaluation metric
        print("Initialize metric for functions")
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask

        return tf.reduce_mean(accuracy_all)
