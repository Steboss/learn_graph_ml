# here we are defining the funciton fw and gw as neural networks
import tensorflow as tf
import numpy as np


class Net:
    r""" In this class we are defining the state and output functions (fw and gw respectively)"""

    def __init__(self, input_dim, state_dim, output_dim):
        r"""Constructor
        Parameters
        ----------
        input_dim: input dimension of the graph, in our case is 70
        state_dim: this is up to the user, default 2
        output_dim: the output must be the same as the labels, in our case 4
        """
        # define an epsilon variable for the iteration procedure
        self.EPSILON = 0.00000001
        # input & output
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_input = self.input_dim - 1 + state_dim  # removing the id_ dimension

        ####  These values are exactly the same as in the official repo,
        # you have to play a bit with this number to find the right value
        self.state_l1 = 5
        self.state_l2 = self.state_dim

        self.output_l1 = 5
        self.output_l2 = self.output_dim


    def netSt(self, inp):
        r""" Define the state function fw
        this is a Dense neural network, with input_dim number of nodes
        Parameters
        ----------
        inp: self.input_dim
        """

        with tf.compat.v1.variable_scope('State_net'):
            # tanh activation function, 2 dense layers
            layer1 = tf.compat.v1.layers.dense(inp, self.state_l1, activation=tf.nn.tanh)
            layer2 = tf.compat.v1.layers.dense(layer1, self.state_l2, activation=tf.nn.tanh)

            return layer2

    def netOut(self, inp):
        r""" Define the output function gw
        same as fw, two dense neural networks with input_dim number of nodes
        Parameters
        ----------
        inp: self.input_dim
        """
        with tf.compat.v1.variable_scope('Output_net'):
            layer1 = tf.compat.v1.layers.dense(inp, self.output_l1, activation=tf.nn.tanh)
            layer2 = tf.compat.v1.layers.dense(layer1, self.output_l2, activation=tf.nn.softmax)

        return layer2

    def Loss(self, output, target, output_weight=None, mask=None):
        r""" loss function for the neural network structure
        """
        # TODO CHECK
        # method to define the loss function
        output = tf.maximum(output, self.EPSILON, name="Avoiding_explosions")  # to avoid explosions
        xent = -tf.reduce_sum(target * tf.math.log(output), 1)

        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        xent *= mask
        lo = tf.reduce_mean(xent)
        return lo

    def Metric(self, target, output, output_weight=None, mask=None):
        # method to define the evaluation metric

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask

        return tf.reduce_mean(accuracy_all)
