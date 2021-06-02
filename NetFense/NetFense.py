import numpy as np
import scipy.sparse as sp
from NetFense.utils import *
from numba import jit


class NetFense:
    def __init__(self, adj, X_obs, z_obs, W1, W2, z_obsp, W1p, W2p, u, verbose=False, AE_par = 0.5, ME_par = 2):
        # Adjacency matrix
        self.AE_par = AE_par
        self.ME_par = ME_par
        self.adj = adj.copy().tolil()
        self.adj_no_selfloops = self.adj.copy()
        self.adj_no_selfloops.setdiag(0)
        self.adj_orig = self.adj.copy().tolil()
        self.u = u  # the node being attacked
        self.adj_preprocessed = preprocess_graph(self.adj).tolil()
        # Number of nodes
        self.N = adj.shape[0]

        # Node attributes
        self.X_obs = X_obs.copy().tolil()
        self.X_obs_orig = self.X_obs.copy().tolil()
        # Node labels
        self.z_obs = z_obs.copy()
        self.z_obsp = z_obsp.copy()

        self.label_u = self.z_obs[self.u]
        self.label_up = self.z_obsp[self.u]

        self.K = np.max(self.z_obs)+1
        self.Kp = np.max(self.z_obsp)+1
        # GCN weight matrices
        self.W1 = W1
        self.W2 = W2
        self.W1p = W1p
        self.W2p = W2p

        self.W = sp.csr_matrix(self.W1.dot(self.W2))
        self.Wp = sp.csr_matrix(self.W1p.dot(self.W2p))

        self.cooc_matrix = self.X_obs.T.dot(self.X_obs).tolil()
        self.cooc_constraint = None

        self.structure_perturbations = []
        # self.feature_perturbations = []

        self.influencer_nodes = []
        self.potential_edges = []
        self.verbose = verbose

    def compute_logits(self):#np.array, [N, K]            The log probabilities for each node.
        return self.adj_preprocessed.dot(self.adj_preprocessed).dot(self.X_obs.dot(self.W))[self.u].toarray()[0]

    def strongest_wrong_class(self, logits): # [N, K]->np.array, [N, L] 
        #The indices of the wrong labels with the highest attached log probabilities.
        label_u_onehot = np.eye(self.K)[self.label_u]
        return (logits - 1000*label_u_onehot).argmax()


    def struct_score(self, a_hat_uv, XW): #

        logits = a_hat_uv.dot(XW)
        logitsp = a_hat_uv.dot(self.compute_XWp())

        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        best_private_class_logits = (logitsp).max(1)
        logits_for_correct_class = logits[:,self.label_u]
        logits_for_correct_private_class = logitsp[:,self.label_up]
        attack_effect = np.abs(logits_for_correct_class - best_wrong_class_logits)#-> small  np.abs
        maitain_effect = (logits_for_correct_private_class ).toarray()#-> large
        struct_scores = np.power(attack_effect, self.AE_par)/np.power(maitain_effect, self.ME_par)
        return struct_scores

    def compute_XW(self):
        return self.X_obs.dot(self.W)
    def compute_XWp(self):
        return self.X_obs.dot(self.Wp)
    def get_attacker_nodes(self, n=5, add_additional_nodes = False):
        assert n < self.N-1, "number of influencers cannot be >= number of nodes in the graph!"

        neighbors = self.adj_no_selfloops[self.u].nonzero()[1]
        assert self.u not in neighbors

        potential_edges = np.column_stack((np.tile(self.u, len(neighbors)),neighbors)).astype("int32")

        # The new A_hat_square_uv values that we would get if we removed the edge from u to each of the neighbors,
        # respectively
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges)

        XW = self.compute_XW()

        # compute the struct scores for all neighbors
        struct_scores = self.struct_score(a_hat_uv, XW).A1

        if len(neighbors) >= n:  # do we have enough neighbors for the number of desired influencers?
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:
            influencer_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, i.e. all nodes except the ones
                # that are already connected to u.
                poss_add_infl = np.setdiff1d(np.arange(self.N), neighbors)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n-len(neighbors)
                possible_edges = np.column_stack((np.tile(self.u, n_possible_additional), poss_add_infl))

                # Compute the struct_scores for all possible additional influencers, and choose the one
                # with the best struct score.
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges)
                additional_struct_scores = self.struct_score(a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(additional_struct_scores)[-n_additional_attackers::]]

                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes

    def compute_new_a_hat_uv(self, potential_edges):

        edges = np.array(self.adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_preprocessed @ self.adj_preprocessed
        values_before = A_hat_sq[self.u].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees,
                                         potential_edges, self.u)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), self.N])

        return a_hat_uv

    def attack_surrogate(self, n_perturbations, delta_cutoff=0.004):
        perturb_features=False
        direct=True
        n_influencers=1
        assert n_perturbations > 0, "need at least one perturbation"

        logits_start = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits_start)
        surrogate_losses = [logits_start[self.label_u] - logits_start[best_wrong_class]]

        if self.verbose:
            print("##### Starting perturbations #####")
            print("##### Perturbate node with ID {} #####".format(self.u))
            print("##### Performing {} perturbations #####".format(n_perturbations))
        influencers = [self.u]
        self.potential_edges = np.column_stack((np.tile(self.u, self.N-1), np.setdiff1d(np.arange(self.N), self.u)))
        self.influencer_nodes = np.array(influencers)
        self.potential_edges = self.potential_edges.astype("int32")
        for _ in range(n_perturbations):
            if self.verbose:
                print("##### ...{}/{} perturbations ... #####".format(_+1, n_perturbations))
            # (perturb_structure) Do not consider edges that, if removed, result in singleton edges in the graph.
            singleton_filter = filter_singletons(self.potential_edges, self.adj)
            filtered_edges = self.potential_edges[singleton_filter]

            filtered_edges_u = filtered_edges[:, 0]
            filtered_edges_v = filtered_edges[:, 1]
            PPR_filter = self.total_change_sym_abs[(filtered_edges_u, filtered_edges_v)]
            filtered_edges_final = filtered_edges[PPR_filter < self.PPR_delta]
            self.filtered_edges_final = filtered_edges_final

            if filtered_edges_final.shape[0]:
              # Compute new entries in A_hat_square_uv
              a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final)
              # Compute the struct scores for each potential edge
              struct_scores = self.struct_score(a_hat_uv_new, self.compute_XW())
              best_edge_ix = struct_scores.argmin()
              best_edge_score = struct_scores.min()
              best_edge = filtered_edges_final[best_edge_ix]
              print("Edge perturbation: {}".format(best_edge))
              if _>0 and self.best[0]==best_edge[0] and self.best[1]==best_edge[1]:
                break
              self.best = best_edge
              change_structure = 1
            else:
              change_structure = 0
              print("No Edge cadicates:")

            if 'change':# change:
              if change_structure:
                  # perform edge perturbation
                  self.adj[tuple(best_edge)] = self.adj[tuple(best_edge[::-1])] = 1 - self.adj[tuple(best_edge)]
                  self.adj_preprocessed = preprocess_graph(self.adj)

                  self.structure_perturbations.append(tuple(best_edge))
                  surrogate_losses.append(best_edge_score)
              else:
                  self.X_obs[tuple(best_feature_ix)] = 1 - self.X_obs[tuple(best_feature_ix)]
                  self.structure_perturbations.append(())
                  surrogate_losses.append(best_feature_score)

    def reset(self):
        """
        Reset 
        """
        self.adj = self.adj_orig.copy()
        self.X_obs = self.X_obs_orig.copy()
        self.structure_perturbations = []
        # self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None


def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before

def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    Parameters
    ----------
    edge_ixs: np.array, shape [E,2], where E is the number of edges in the graph.
        The indices of the nodes connected by the edges in the input graph.
    node_nb_ixs: np.array, shape [N,], dtype int
        For each node, this gives the first index of edges associated to this node in the edge array (edge_ixs).
        This will be used to quickly look up the neighbors of a node, since numba does not allow nested lists.
    edges_set: set((e0, e1))
        The set of edges in the input graph, i.e. e0 and e1 are two nodes connected by an edge
    twohop_ixs: np.array, shape [T, 2], where T is the number of edges in A_tilde^2
        The indices of nodes that are in the twohop neighborhood of each other, including self-loops.
    values_before: np.array, shape [N,], the values in [A_hat]^2_uv to be updated.
    degs: np.array, shape [N,], dtype int
        The degree of the nodes in the input graph.
    potential_edges: np.array, shape [P, 2], where P is the number of potential edges.
        The potential edges to be evaluated. For each of these potential edges, this function will compute the values
        in [A_hat]^2_uv that would result after inserting/removing this edge.
    u: int
        The target node

    Returns
    -------
    return_ixs: List of tuples
        The ixs in the [P, N] matrix of updated values that have changed
    return_values:

    """
    N = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values

def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.

    Parameters
    ----------
    edges: np.array, shape [P, 2], dtype int, where P is the number of input edges.
        The potential edges.

    adj: sp.sparse_matrix, shape [N,N]
        The input adjacency matrix.

    Returns
    -------
    np.array, shape [P, 2], dtype bool:
        A binary vector of length len(edges), False values indicate that the edge at
        the index  generates singleton edges, and should thus be avoided.

    """

    degs = np.squeeze(np.array(np.sum(adj,0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1

    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0

