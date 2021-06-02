import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #0
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
from tensorflow.contrib import slim
import numpy as np
from sklearn.metrics import f1_score
from NetFense.utils import *

spdot =  tf.sparse.sparse_dense_matmul#tf.sparse_tensor_dense_matmul
dot = tf.matmul

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor +=  tf.random.uniform(noise_shape)# tf.random_uniform
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out =  tf.sparse.retain(x, dropout_mask)# tf.sparse_retain
    return pre_out * (1./keep_prob)

class GCN:
    def __init__(self, sizes, An, X_obs, name="", with_relu=True, params_dict={'dropout': 0.5}, gpu_id=0,
                 seed=-1):
        """
        Create a Graph Convolutional Network model in Tensorflow with one hidden layer.

        Parameters
        ----------
        sizes: list
            List containing the hidden and output sizes (i.e. number of classes). E.g. [16, 7]

        An: sp.sparse_matrix, shape [N,N]
            The input adjacency matrix preprocessed using the procedure described in the GCN paper.

        X_obs: sp.sparse_matrix, shape [N,D]
            The node features.

        name: string, default: ""
            Name of the network.

        with_relu: bool, default: True
            Whether there a nonlinear activation function (ReLU) is used. If False, there will also be
            no bias terms, no regularization and no dropout.

        params_dict: dict
            Dictionary containing other model parameters.

        gpu_id: int or None, default: 0
            The GPU ID to be used by Tensorflow. If None, CPU will be used

        seed: int, defualt: -1
            Random initialization for reproducibility. Will be ignored if it is -1.
        """

        self.graph = tf.Graph()
        if seed > -1:
            tf.set_random_seed(seed)

        if An.format != "csr":
            An = An.tocsr()

        with self.graph.as_default():

            with  tf.compat.v1.variable_scope(name) as scope:#tf.variable_scope
                w_init = slim.xavier_initializer
                self.name = name
                self.n_classes = sizes[1]

                self.dropout = params_dict['dropout'] if 'dropout' in params_dict else 0.
                if not with_relu:
                    self.dropout = 0

                self.learning_rate = params_dict['learning_rate'] if 'learning_rate' in params_dict else 0.01

                self.weight_decay = params_dict['weight_decay'] if 'weight_decay' in params_dict else 5e-4
                self.N, self.D = X_obs.shape

                self.node_ids = tf.compat.v1.placeholder(tf.int32, [None], 'node_ids')# tf.placeholder
                self.node_labels = tf.compat.v1.placeholder(tf.int32, [None, sizes[1]], 'node_labels')# tf.placeholder

                # bool placeholder to turn on dropout during training
                self.training = tf.compat.v1.placeholder_with_default(False, shape=())# tf.placeholder_with_default

                self.An = tf.SparseTensor(np.array(An.nonzero()).T, An[An.nonzero()].A1, An.shape)
                self.An = tf.cast(self.An, tf.float32)
                self.X_sparse = tf.SparseTensor(np.array(X_obs.nonzero()).T, X_obs[X_obs.nonzero()].A1, X_obs.shape)
                self.X_dropout = sparse_dropout(self.X_sparse, 1 - self.dropout,
                                                (int(self.X_sparse.values.get_shape()[0]),))
                # only use drop-out during training
                self.X_comp = tf.cond(self.training,
                                      lambda: self.X_dropout,
                                      lambda: self.X_sparse) if self.dropout > 0. else self.X_sparse

                self.W1 = slim.variable('W1', [self.D, sizes[0]], tf.float32, initializer=w_init())
                self.b1 = slim.variable('b1', dtype=tf.float32, initializer=tf.zeros(sizes[0]))

                self.h1 = spdot(self.An, spdot(self.X_comp, self.W1))

                if with_relu:
                    self.h1 = tf.nn.relu(self.h1 + self.b1)

                self.h1_dropout = tf.nn.dropout(self.h1, rate = self.dropout)# 1 - self.dropout


                self.h1_comp = tf.cond(self.training,
                                       lambda: self.h1_dropout,
                                       lambda: self.h1) if self.dropout > 0. else self.h1

                self.W2 = slim.variable('W2', [sizes[0], sizes[1]], tf.float32, initializer=w_init())
                self.b2 = slim.variable('b2', dtype=tf.float32, initializer=tf.zeros(sizes[1]))

                self.logits = spdot(self.An, dot(self.h1_comp, self.W2))
                if with_relu:
                    self.logits += self.b2
                self.logits_gather = tf.gather(self.logits, self.node_ids)

                self.predictions = tf.nn.softmax(self.logits_gather)

                self.loss_per_node = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_gather,
                                                                             labels=self.node_labels)#tf.nn.softmax_cross_entropy_with_logits
                self.loss = tf.reduce_mean(self.loss_per_node)

                # weight decay only on the first layer, to match the original implementation
                if with_relu:
                    self.loss += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in [self.W1, self.b1]])

                var_l = [self.W1, self.W2]
                if with_relu:
                    var_l.extend([self.b1, self.b2])
                self.train_op =  tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                  var_list=var_l)# tf.train.AdamOptimizer

                self.varlist =  tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)# tf.get_collection #tf.GraphKeys.GLOBAL_VARIABLES
                self.local_init_op =  tf.compat.v1.variables_initializer(self.varlist)# tf.variables_initializer

                if gpu_id is None:
                    config =  tf.compat.v1.ConfigProto(
                        device_count={'GPU': 0}
                    )# tf.ConfigProto
                else:
                    gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                    config = tf.ConfigProto(gpu_options=gpu_options)

                self.session = tf.compat.v1.InteractiveSession(config=config)#tf.InteractiveSession
                self.init_op = tf.compat.v1.global_variables_initializer()# tf.global_variables_initializer
                self.session.run(self.init_op)

    def convert_varname(self, vname, to_namespace=None):
        """
        Utility function that converts variable names to the input namespace.

        Parameters
        ----------
        vname: string
            The variable name.

        to_namespace: string
            The target namespace.

        Returns
        -------

        """
        namespace = vname.split("/")[0]
        if to_namespace is None:
            to_namespace = self.name
        return vname.replace(namespace, to_namespace)

    def set_variables(self, var_dict):
        """
        Set the model's variables to those provided in var_dict. This is e.g. used to restore the best seen parameters
        after training with patience.

        Parameters
        ----------
        var_dict: dict
            Dictionary of the form {var_name: var_value} to assign the variables in the model.

        Returns
        -------
        None.
        """

        with self.graph.as_default():
            if not hasattr(self, 'assign_placeholders'):
                self.assign_placeholders = {v.name: tf.compat.v1.placeholder(v.dtype, shape=v.get_shape()) for v in self.varlist}# tf.placeholder
                self.assign_ops = {v.name:  tf.compat.v1.assign(v, self.assign_placeholders[v.name])
                                   for v in self.varlist}# tf.assign
            to_namespace = list(var_dict.keys())[0].split("/")[0]
            self.session.run(list(self.assign_ops.values()), feed_dict = {val: var_dict[self.convert_varname(key, to_namespace)]
                                                                     for key, val in self.assign_placeholders.items()})

    def train(self, split_train, split_val, Z_obs, patience=30, n_iters=200, print_info=True):
        """
        Train the GCN model on the provided data.

        Parameters
        ----------
        split_train: np.array, shape [n_train,]
            The indices of the nodes used for training

        split_val: np.array, shape [n_val,]
            The indices of the nodes used for validation.

        Z_obs: np.array, shape [N,k]
            All node labels in one-hot form (the labels of nodes outside of split_train and split_val will not be used.

        patience: int, default: 30
            After how many steps without improvement of validation error to stop training.

        n_iters: int, default: 200
            Maximum number of iterations (usually we hit the patience limit earlier)

        print_info: bool, default: True

        Returns
        -------
        None.

        """

        varlist = self.varlist
        self.session.run(self.local_init_op)
        early_stopping = patience

        best_performance = 0
        patience = early_stopping

        feed = {self.node_ids: split_train,
                self.node_labels: Z_obs[split_train]}
        if hasattr(self, 'training'):
            feed[self.training] = True
        for it in range(n_iters):
            _loss, _ = self.session.run([self.loss, self.train_op], feed)
            f1_micro, f1_macro = eval_class(split_val, self, np.argmax(Z_obs, 1))
            perf_sum = f1_micro + f1_macro
            if perf_sum > best_performance:
                best_performance = perf_sum
                patience = early_stopping
                # var dump to memory is much faster than to disk using checkpoints
                var_dump_best = {v.name: v.eval(self.session) for v in varlist}
            else:
                patience -= 1
            if it > early_stopping and patience <= 0:
                break
        if print_info:
            print('converged after {} iterations'.format(it - patience))
        # Put the best observed parameters back into the model
        self.set_variables(var_dump_best)


def pre_train(split_train, split_val, _Z_obs, _Z_obsp, sizes, sizesp, _An, _X_obs, gpu_id):

    _N = _An.shape[0]
    print('Pretrain...Target CL')
    surrogate_model = GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
    surrogate_model.train(split_train, split_val, _Z_obs, print_info=False)
    W1 =surrogate_model.W1.eval(session=surrogate_model.session)
    W2 =surrogate_model.W2.eval(session=surrogate_model.session)
    pred = eval_model_confidence(surrogate_model, np.arange(_N), split_val, split_train, _Z_obs)
    # print(pred)
    # print(_Z_obs[0])
    # print(np.round(surrogate_model.predictions.eval(session=surrogate_model.session,feed_dict={surrogate_model.node_ids: [0]})[0], 3))
    surrogate_model.session.close()

    surrogate_modelp = GCN(sizesp, _An, _X_obs, with_relu=False, name="surrogatep", gpu_id=gpu_id)
    surrogate_modelp.train(split_train, split_val, _Z_obsp, print_info=False)
    print('Pretrain...Privacy CL')
    W1p =surrogate_modelp.W1.eval(session=surrogate_modelp.session)
    W2p =surrogate_modelp.W2.eval(session=surrogate_modelp.session)
    # print(_Z_obsp[0])
    # print(np.round(surrogate_modelp.predictions.eval(session=surrogate_modelp.session,feed_dict={surrogate_modelp.node_ids: [0]})[0], 3))
    # print(eval_model_confidence(surrogate_modelp, np.arange(_N), split_val, split_train, _Z_obsp))
    surrogate_modelp.session.close()

    return pred, W1, W2, W1p, W2p

def eval_class(ids_to_eval, model, z_obs):
    """
    Evaluate the model's classification performance.

    Parameters
    ----------
    ids_to_eval: np.array
        The indices of the nodes whose predictions will be evaluated.

    model: GCN
        The model to evaluate.

    z_obs: np.array
        The labels of the nodes in ids_to_eval

    Returns
    -------
    [f1_micro, f1_macro] scores

    """
    test_pred = model.predictions.eval(session=model.session, feed_dict={model.node_ids: ids_to_eval}).argmax(1)
    test_real = z_obs[ids_to_eval]

    return f1_score(test_real, test_pred, average='micro'), f1_score(test_real, test_pred, average='macro')
