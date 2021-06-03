from NetFense.utils import *
from NetFense.GCN import *
from NetFense.NetFense import *
import numpy as np
import scipy.sparse as sp
import os 
import argparse
import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2) #pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')#return pickle.load(f)


def parse_args():
    #Parses the arguments.
    parser = argparse.ArgumentParser(description="CoANE")
    parser.add_argument('--data_name', type=str, default='citeseer', help='Name of Dataset citeseer/cora/TerroristRel for Citeseer/Cora/PIT')
    parser.add_argument('--verbose', type=bool, default=True, help='Print log')
    
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id ')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_hid', type=int, default=16, help='hidden size of GCN')
    parser.add_argument('--tar', type=int, default=1132, help='target node')

    parser.add_argument('--test_frac', type=float, default=.8, help='Split val data ratio')
    parser.add_argument('--val_frac', type=float, default=.1, help='Split val data ratio')
    parser.add_argument('--AE_par', type=float, default=1, help='Perturbation par')
    parser.add_argument('--ME_par', type=float, default=2, help='Maintenance par')#2

    return parser.parse_args() 

def split(args, _N, _z_obs):
    unlabeled_share = args.test_frac
    val_share = args.val_frac
    train_share = 1 - unlabeled_share - val_share

    np.random.seed(15)
    split_train, split_val, split_unlabeled = train_val_test_split_tabular(np.arange(_N),
                                                                           train_size=train_share,
                                                                           val_size=val_share,
                                                                           test_size=unlabeled_share,
                                                                           stratify=_z_obs,
                                                                           random_state=args.seed)

    return split_train, split_val, split_unlabeled

def Evaluation_train(adj, Z_l, z_l, u, size,  X, gpu_id, r, s_t, s_v):
    margins_ = []
    gcn_ = GCN(size, adj, X, "gcn_orig", gpu_id=gpu_id) # gcn_before
    for _ in range(r):
        gcn_.train(s_t, s_v, Z_l, print_info=False)
        probs_before_attack = gcn_.predictions.eval(session=gcn_.session,feed_dict={gcn_.node_ids: [u]})[0]
        best_second_class_before = (probs_before_attack - 1000*Z_l[u]).argmax()
        margin_before = probs_before_attack[z_l[u]] - probs_before_attack[best_second_class_before]
        margins_.append(margin_before)
    gcn_.session.close()
    print(margins_[-1])#np.round(probs_before_attack, 3),  Z_l.shape, 
    print('Ground Truth Label: ', z_l[u])
    return probs_before_attack, margins_


def run():
    args = parse_args()
    gpu_id = None # set this to your desired GPU ID if you want to use GPU computations 
    data_name = args.data_name
    n_hid = args.n_hid
    AE_par = args.AE_par
    ME_par = args.ME_par
    print("AE_par, ME_par: ", AE_par, ME_par)

    # 'load data and set feat_choose':
    _X_obs, _A_obs, _An, labels, PPR_delta, total_change_sym_abs, _z_obs, _z_obsp, _Z_obs, _Z_obsp  = load_data(data_name)
    _N = _A_obs.shape[0]
    _K = _z_obs.max()+1
    _Kp = _z_obsp.max()+1
    sizes = [n_hid, _K]
    sizesp = [n_hid, _Kp]
    degrees = _A_obs.sum(0).A1

    if 'split and pre-trained model':
        print('\nSplit and pre-trained model......')
        split_train, split_val, split_unlabeled = split(args, _N, _z_obs)
        print('\nPre-train............')
        pred, W1, W2, W1p, W2p = pre_train(split_train, split_val, _Z_obs, _Z_obsp, sizes, sizesp, _An, _X_obs, gpu_id)
        print('......')

        _z_obs_pred = _z_obs
        _z_obs_pred[split_unlabeled] = pred[0][split_unlabeled]

    print('\nRun Experiment......')

    ixdd = 0
    u = args.tar
    assert u in split_unlabeled
    NF = NetFense(_A_obs, _X_obs, _z_obsp, W1p, W2p, _z_obs_pred, W1, W2, u, verbose=False, AE_par = AE_par, ME_par = ME_par)
    NF.PPR_delta = PPR_delta
    NF.total_change_sym_abs = total_change_sym_abs
    
    n_perturbations = np.max((np.min((int(degrees[u]), 20)),1))

    NF.reset()
    print('n_perturbations: ', n_perturbations)
    NF.attack_surrogate(n_perturbations)
    retrain_iters=5

    # # CL
    print('\nRe-train...Target CL')
    print('\nBefore Maring')
    _, _ = Evaluation_train(_An, _Z_obs, _z_obs, NF.u, sizes,  _X_obs, gpu_id, retrain_iters, split_train, split_val)

    print('\nAfter Maring')
    _, _ = Evaluation_train(NF.adj_preprocessed, _Z_obs, _z_obs, NF.u, sizes,  NF.X_obs.tocsr(), gpu_id, retrain_iters, split_train, split_val)

    # privacy CL
    print('\nRe-train...Private CL')
    print('\nBefore Maring')
    _, _ = Evaluation_train(_An, _Z_obsp, _z_obsp, NF.u, sizesp,  _X_obs, gpu_id, retrain_iters, split_train, split_val)

    print('\nAfter Maring')
    _, _ = Evaluation_train(NF.adj_preprocessed, _Z_obsp, _z_obsp, NF.u, sizesp,  NF.X_obs.tocsr(), gpu_id, retrain_iters, split_train, split_val)

    print('\n')


if __name__ == "__main__":
    run()
