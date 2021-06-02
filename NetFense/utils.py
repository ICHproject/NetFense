from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components

def load_data(data_name):
    if data_name == 'citeseer' or data_name == 'cora':
        file_name = 'data/{}.npz'.format(data_name)
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        if not file_name.endswith('.npz'):
                file_name += '.npz'
        with np.load(file_name) as loader:
                loader = dict(loader)
                adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                                                                            loader['adj_indptr']), shape=loader['adj_shape'])

                if 'attr_data' in loader:
                        attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                                                                        loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                        attr_matrix = None
                labels = loader.get('labels')
    elif data_name == 'BlogCatalog':
        attr_matrix, adj_matrix, labels = load_obj('data/BlogCatalog')
    elif data_name == 'TerroristRel':
        attr_matrix, adj_matrix, labels, labels2, labels3, labels4 = load_obj('data/TerroristRel_')

    if data_name == 'citeseer':
        feat_choose = 65
    elif data_name == 'cora':
        feat_choose = 1177
    elif data_name == 'BlogCatalog':
        feat_choose = 15
    elif data_name == 'TerroristRel':
        feat_choose = 744  

    _An, _A_obs, _X_obs, _z_obs, _z_obsp, _Z_obs, _Z_obsp = preprocess_lcc(attr_matrix, adj_matrix, labels, feat_choose)

    if 'load PPR':
        try:
          total_change_sym_abs = load_obj('PPR_{0}'.format(data_name))
        except:
          total_change_sym_abs = compute_PPR_influence(_A_obs)

        PPR_IF_tri = total_change_sym_abs[np.triu_indices(total_change_sym_abs.shape[0], 1)]
        PPR_delta = np.quantile(PPR_IF_tri, 0.9)
    
    return _X_obs, _A_obs, _An, labels, PPR_delta, total_change_sym_abs, _z_obs, _z_obsp, _Z_obs, _Z_obsp 

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
            Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
            Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
            file_name += '.npz'
    with np.load(file_name) as loader:
            loader = dict(loader)
            adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                                                                        loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                    attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                                                                 loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                    attr_matrix = None

            labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
            Input graph.
    n_components : int, default 1
            Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
            Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
                    Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
            Proportion of the dataset included in the train split.
    val_size : float, default 0.3
            Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
            Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
            If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
            Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
            List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
            raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                                                                 random_state=random_state,
                                                                                                 train_size=(train_size + val_size),
                                                                                                 test_size=test_size,
                                                                                                 stratify=stratify)
    if stratify is not None:
            stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                                                                random_state=random_state,
                                                                                train_size=(train_size / (train_size + val_size)),
                                                                                test_size=(val_size / (train_size + val_size)),
                                                                                stratify=stratify)
    result = []
    for X in arrays:
            result.append(X[idx_train])
            result.append(X[idx_val])
            result.append(X[idx_test])
    return result

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized

def compute_PPR_influence(adj):
    D = adj.toarray().sum(-1)
    D_ = np.eye(D.shape[0])/D
    Ieye = (np.eye(adj.shape[0]))
    al = 0.1
    #PPR original 
    M_1 = np.linalg.inv(Ieye- (1-al)*(D_@adj.toarray()))
    #compute sum of change of PPR
    adj = adj.toarray()
    Pc = M_1.sum(0)
    #up and down
    Change_up = (2*adj-np.ones(adj.shape))*Pc.reshape((-1,1))/D.reshape((-1,1))
    Change_down = np.ones(adj.shape) - (2*adj-np.ones(adj.shape))*(1-al)*(D_@M_1)*(1)
    total_change = Change_up/Change_down
    total_change_sym = (total_change+total_change.T)
    total_change_sym_abs = np.abs(total_change+total_change.T)
    return total_change_sym_abs


def eval_model_confidence(e_mode, unlabeled, val, train, Z):
    pred = e_mode.predictions.eval(session=e_mode.session, feed_dict={e_mode.node_ids: unlabeled})
    return [pred.argmax(1) , pred.max(1)]

def preprocess_lcc(_X_obs, _A_obs, labels, feat_choose):
    # _X_obs, _A_obs, labels, feat_choose = load_data(data_name)

    _z_obs = labels
    _z_obsp = (_X_obs.toarray()[:,feat_choose]).astype('int')#private labels
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1

    # choose complete part
    lcc = largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc][:,lcc]
    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"
    _X_obs = _X_obs[lcc].astype('float32')
    _z_obs = _z_obs[lcc]
    _z_obsp = _z_obsp[lcc]

    _K = _z_obs.max()+1
    _Kp = _z_obsp.max()+1

    _Z_obs = np.eye(_K)[_z_obs]
    _Z_obsp = np.eye(_Kp)[_z_obsp]
    _An = preprocess_graph(_A_obs)

    return _An, _A_obs, _X_obs, _z_obs, _z_obsp, _Z_obs, _Z_obsp