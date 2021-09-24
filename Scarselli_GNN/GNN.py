import tensorflow as tf
import numpy as np
import datetime as time

# class for the core of the architecture
class GNN:
    def __init__(self,
                 net,
                 input_dim,
                 output_dim,
                 state_dim,
                 max_it=50,
                 optimizer=tf.compat.v1.train.AdamOptimizer,
                 learning_rate=0.01,
                 threshold=0.01,
                 graph_based=False,
                 param=str(time.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
                 config=None,
                 tensorboard=False,
                 mask_flag=False):
        """
        Constructor
        Initialize all the elements of the graph neural networks

        Parameters
        -----------
        net: Net instance - it contains state network, output network, initialized weights, loss function and metric;
        input_dim: dimension of the input
        output_dim: dimension of the output
        state_dim: dimension for the state
        max_it:  maximum number of iteration of the state convergence procedure
        optimizer:  optimizer instance
        learning_rate: learning rate value
        threshold:  value to establish the state convergence
        graph_based: flag to denote a graph based problem
        param: name of the experiment
        config: ConfigProto protocol buffer object, to set configuration options for a session
        tensorboard:  boolean flag to activate tensorboard
        mask_flag:  boolean flag to activate semisupervised
        """

        np.random.seed(0)
        tf.random.set_seed(0)
        print("Tensorboard ", tensorboard)
        self.tensorboard = tensorboard
        print("Max iteration number ", max_it)
        self.max_iter = max_it
        self.net = net
        self.optimizer = optimizer(learning_rate, name="optim")
        print("Threshold ", threshold)
        self.state_threshold = threshold
        print("input dimension ", input_dim)
        self.input_dim = input_dim
        print("output dimension ", output_dim)
        self.output_dim = output_dim
        print("state dimension ", state_dim)
        self.state_dim = state_dim
        self.graph_based = graph_based
        self.mask_flag = mask_flag
        self.build()

        self.session = tf.compat.v1.Session(config=config)
        self.session.run(tf.compat.v1.global_variables_initializer())
        self.init_l = tf.compat.v1.local_variables_initializer()

        # parameter to monitor the learning via tensorboard and to save the model
        if self.tensorboard:
            self.merged_all = tf.compat.v1.summary.merge_all(key='always')
            self.merged_train = tf.compat.v1.summary.merge_all(key='train')
            self.merged_val = tf.compat.v1.summary.merge_all(key='val')
            self.writer = tf.compat.v1.summary.FileWriter('tmp/' + param, self.session.graph)


    def VariableState(self):
        r""" This function define placeholders for input, target, state, old_state and arcnode"""

        # placeholder for input and output
        self.comp_inp = tf.compat.v1.placeholder(tf.float32, shape=(None, self.input_dim), name="input")
        self.y = tf.compat.v1.placeholder(tf.float32, shape=(None, self.output_dim), name="target")

        if self.mask_flag: # e.g. train
            self.mask = tf.compat.v1.placeholder(tf.float32, name="mask")

        # state(t) & state(t-1)
        self.state = tf.compat.v1.placeholder(tf.float32, shape=(None, self.state_dim), name="state")
        self.state_old = tf.compat.v1.placeholder(tf.float32, shape=(None, self.state_dim), name="old_state")

        # arch-node conversion matrix
        self.ArcNode = tf.compat.v1.sparse_placeholder(tf.float32, name="ArcNode")

        # node-graph conversion matrix
        if self.graph_based:
            self.NodeGraph = tf.compat.v1.sparse_placeholder(tf.float32, name="NodeGraph")
        else:
            self.NodeGraph = tf.compat.v1.placeholder(tf.float32, name="NodeGraph")

    def build(self):
        r""" Function to build the NN architecture"""

        # network
        self.VariableState()
        self.loss_op = self.Loop()

        # loss
        with tf.compat.v1.variable_scope('loss'):

            if self.mask_flag:
                self.loss = self.net.Loss(self.loss_op[0], self.y, mask=self.mask)
                self.val_loss = self.net.Loss(self.loss_op[0], self.y, mask=self.mask)
            else:
                self.loss = self.net.Loss(self.loss_op[0], self.y)
                # val loss
                self.val_loss = self.net.Loss(self.loss_op[0], self.y)

            if self.tensorboard:
                self.summ_loss = tf.compat.v1.summary.scalar('loss', self.loss, collections=['train'])
                self.summ_val_loss = tf.compat.v1.summary.scalar('val_loss', self.val_loss, collections=['val'])

        # optimizer --> backprop step
        with tf.compat.v1.variable_scope('train'):
            self.grads = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(self.grads, name='train_op')
            if self.tensorboard:
                for index, grad in enumerate(self.grads):
                    tf.compat.v1.summary.histogram("{}-grad".format(self.grads[index][1].name), self.grads[index],
                                         collections=['always'])

        # metrics
        with tf.compat.v1.variable_scope('metrics'):
            if self.mask_flag:
                self.metrics = self.net.Metric(self.y, self.loss_op[0], mask=self.mask)
            else:
                self.metrics = self.net.Metric(self.y, self.loss_op[0])

        # val metric
        with tf.compat.v1.variable_scope('val_metric'):
            if self.mask_flag:
                self.val_met = self.net.Metric(self.y, self.loss_op[0], mask=self.mask)
            else:
                self.val_met = self.net.Metric(self.y, self.loss_op[0])
            if self.tensorboard:
                self.summ_val_met = tf.compat.v1.summary.scalar('val_metric', self.val_met, collections=['always'])


    def convergence(self, a, state, old_state, k):
        r""" This is a very important function as it establish if we have reached the
        convergence of the iterative procedure

        Parameters
        ----------
        a: input matrix
        state: newly computed state or randomly assigned state
        old_state: current nodes' state
        k: iteration number
        """
        with tf.compat.v1.variable_scope('Convergence'):
            # body of the while cycle used to iteratively calculate state

            # assign current state to old state
            old_state = state
            # grab states of neighboring node a[:,1] is the input matrix and [:,1] takes all the destination nodes
            gat = tf.gather(old_state, tf.cast(a[:, 1], tf.int32))
            # slice to consider only label of the node and that of its neighbor
            sl = a[:, 2:] # these are the features
            # concat with retrieved state
            inp = tf.concat([sl, gat], axis=1)
            # HERE WE HAVE THE CORE OF GRAPH ML
            # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
            layer1 = self.net.netSt(inp)
            # matrix multiplication between input edges matrix and output of layer 1
            state = tf.compat.v1.sparse_tensor_dense_matmul(self.ArcNode, layer1)
            # update the iteration counter
            k = k + 1
        return a, state, old_state, k

    def condition(self, a, state, old_state, k):
        r""" This function check whether the new state is a fixed-point"""
        # evaluate condition on the convergence of the state
        with tf.compat.v1.variable_scope('condition'):
            # evaluate distance by state(t) and state(t-1)
            outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, old_state)), 1) + 0.00000000001)
            # vector showing item converged or not (given a certain threshold) --> contraction map concept here
            checkDistanceVec = tf.greater(outDistance, self.state_threshold)

            c1 = tf.reduce_any(checkDistanceVec)
            c2 = tf.less(k, self.max_iter)

        return tf.logical_and(c1, c2)


    def Loop(self):
        r""" Loop function is the core of the run
        Here the condition and convergence are checked for each new state
        Once the tf.while_loop has reached convergence:
        - if we need a graph_based prediction we multiply the NodeGraph (archnode matrix form prepare_GNN) with
        the current state
        - otherwise we keep the current state
        and then we feed this in the GW MLP function

        Return
        -------
        out: array: current predictions
        num: iteration number
        stf: current nodes' state"""
        # call to loop for the state computation and compute the output
        # compute state
        with tf.compat.v1.variable_scope('Loop'):

            k = tf.constant(0)
            res, st, old_st, num = tf.while_loop(self.condition,
                                                 self.convergence,
                                                 [self.comp_inp, self.state, self.state_old, k])


            if self.tensorboard:
                self.summ_iter = tf.compat.v1.summary.scalar('iteration',
                                                             num,
                                                             collections=['always'])

            if self.graph_based:
                # stf = tf.transpose(tf.matmul(tf.transpose(st), self.NodeGraph))
                stf = tf.sparse_tensor_dense_matmul(self.NodeGraph, st)
            else:
                stf = st
            # multiply the output of the new state with the output network in each node
            out = self.net.netOut(stf)

        return out, num, stf, res

    def Train(self, inputs, ArcNode, target, step, nodegraph=0.0, mask=None):
        ''' train methods: has to receive the inputs, arch-node matrix conversion, target,
        and optionally nodegraph indicator

        Return
        ------
        loss: current loss
        loop: 3D matrix: loop[0] current prediction, loop[1] current state, loop[2] number of iterations'''

        # Creating a SparseTensor with the feeded ArcNode Matrix
        arcnode_ = tf.compat.v1.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        if self.graph_based:
            nodegraph = tf.compat.v1.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                        dense_shape=nodegraph.dense_shape)

        if self.mask_flag:
            fd = {self.NodeGraph: nodegraph,
                  self.comp_inp: inputs,
                  self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
                  self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
                  self.ArcNode: arcnode_,
                  self.y: target,
                  self.mask: mask}

        else:

            fd = {self.NodeGraph: nodegraph,
                  self.comp_inp: inputs,
                  self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
                  self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
                  self.ArcNode: arcnode_,
                  self.y: target}

        if self.tensorboard:
            _, loss, loop, merge_all, merge_tr = self.session.run(
                [self.train_op,
                 self.loss,
                 self.loss_op,
                 self.merged_all,
                 self.merged_train],
                feed_dict=fd)

            # loss is current loss
            # loop returns the predictions proba for labels
            # e/g/ node[0] = [0.00, 0.00, 0.9, 0.00]

            if step % 100 == 0:
                self.writer.add_summary(merge_all, step)
                self.writer.add_summary(merge_tr, step)
        else:
            _, loss, loop = self.session.run(
                [self.train_op, self.loss, self.loss_op],
                feed_dict=fd)

        # loop[1] return the number of iteration for reaching convergence
        # loop[2] return the new state of the node which reflect the predictions

        return loss, loop

    def Validate(self, inptVal, arcnodeVal, targetVal, step, nodegraph=0.0, mask=None):
        """ Takes care of the validation of the model - it outputs, regarding the set given as input,
         the loss value, the accuracy (custom defined in the Net file), the number of iteration
         in the convergence procedure """

        arcnode_ = tf.compat.v1.SparseTensorValue(indices=arcnodeVal.indices,
                                                  values=arcnodeVal.values,
                                        dense_shape=arcnodeVal.dense_shape)
        if self.graph_based:
            nodegraph = tf.compat.v1.SparseTensorValue(indices=nodegraph.indices,
                                                       values=nodegraph.values,
                                        dense_shape=nodegraph.dense_shape)

        if self.mask_flag:
            fd_val = {self.NodeGraph: nodegraph, self.comp_inp: inptVal,
                      self.state: np.zeros((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.state_old: np.ones((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.ArcNode: arcnode_,
                      self.y: targetVal,
                      self.mask: mask}
        else:

            fd_val = {self.NodeGraph: nodegraph, self.comp_inp: inptVal,
                      self.state: np.zeros((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.state_old: np.ones((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.ArcNode: arcnode_,
                      self.y: targetVal}

        if self.tensorboard:
            loss_val, loop, merge_all, merge_val, metr = self.session.run(
                [self.val_loss,
                 self.loss_op,
                 self.merged_all,
                 self.merged_val,
                 self.metrics
                 ],
                feed_dict=fd_val)
            self.writer.add_summary(merge_all, step)
            self.writer.add_summary(merge_val, step)
        else:
            loss_val, loop, metr = self.session.run(
                [self.val_loss, self.loss_op, self.metrics], feed_dict=fd_val)

        # return loss, accuracy and number of iteration
        return loss_val, metr, loop[1]


    def Evaluate(self, inputs, ArcNode, target, nodegraph=0.0):
        '''evaluate methods: has to receive the inputs,  arch-node matrix conversion, target
         -- gives as output the accuracy on the set given as input'''

        arcnode_ = tf.compat.v1.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        if self.graph_based:
            nodegraph = tf.compat.v1.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                        dense_shape=nodegraph.dense_shape)


        fd = {self.NodeGraph: nodegraph,
              self.comp_inp: inputs,
              self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
              self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
              self.ArcNode: arcnode_, self.y: target}
        _ = self.session.run([self.init_l])
        met = self.session.run([self.metrics], feed_dict=fd)
        return met


    def Predict(self, inputs, ArcNode, nodegraph=0.0):
        ''' predict methods: has to receive the inputs, arch-node matrix conversion -- gives as output the output
         values of the output function (all the nodes output
         for all the graphs (if node-based) or a single output for each graph (if graph based) '''

        arcnode_ = tf.compat.v1.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        fd = {self.comp_inp: inputs, self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
              self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
              self.ArcNode: arcnode_}
        pr = self.session.run([self.loss_op], feed_dict=fd)
        return pr[0]