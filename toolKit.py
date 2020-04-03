import sys
sys.path.append("./")
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
import json
import scipy.sparse
from sklearn.metrics import classification_report, accuracy_score
import svd_new
from itertools import combinations

def load_graph(data_path):

    """
    Load the GC order graph between givers and claimers.
    Parameters:
        data_path: the storage path of text file in which each line refers to one edge, (start-point, end-point).   
    returns: 
        giver_2_claimer (dict): key is one specific giver and value is the list of claimers associated with this giver.
        claimer_2_giver (dict): key is one specific claimer and value is the list of givers associated with this claimer.
    """
    
    with open(data_path, "r") as f_graph:
        # f_graph.readline()
        giver_2_claimer = defaultdict(list)
        claimer_2_giver = defaultdict(list)
        for line in f_graph.readlines():
            _giver = int(line.strip().split(" ")[0].strip())
            _claimer = int(line.strip().split(" ")[1].strip())
            giver_2_claimer[_giver].append(_claimer)
            claimer_2_giver[_claimer].append(_giver)
            
    return giver_2_claimer, claimer_2_giver

def component_adjacency_matrix(G, mat):
    """
    Get the adjecency matrix associated with one subgraph

    Parameters: 
        G (a graph instance in networkx): a subgraph
        mat (csc matrix): an adjecency matrix assoicated with the subgraph
    Returns: 
        mat_data: the adjecency matrix of graph G.
        _givers: the list of giver customers in GC graph G.
        _claimers: the list of claimer customers in GC graph G.
    """

    _givers = []
    _claimers = []
    for _edge in G.edges:
        _givers += [_edge[0]]
        _claimers += [_edge[1]]
    _givers = list(set(_givers))
    _claimers = list(set(_claimers))
    mat_data = []
    for _gid in _givers:
        mat_data.append([mat[_gid, _cid] for _cid in _claimers])
    return np.array(mat_data), _givers, _claimers


def component_heatmap(_mat_data, _givers, _claimers, link_method):
    """ 
    Draw the heatmap of GC transactions associated with customers lying in one particular GC component. 

    Parameters: 
        _mat_data: the weighted adjecency matrix assocated with customers in one specific component
        _givers: the list of giver customers in one particular component.
        _claimers: the list of claimer customers in one certain component.
        link_method: the linkage method for hierarchical clustering.
    returns:
        Output the heatmap of GC transactions associated with customers given and weight info. attached.
    """

    mat_data_1 = np.asarray(list(_mat_data))
    Y = sch.linkage(mat_data_1, method=link_method)
    idx1 = sch.dendrogram(Y)['leaves']

    mat_data_trans = mat_data_1.transpose()
    Y = sch.linkage(mat_data_trans, method=link_method)
    idx2 = sch.dendrogram(Y)['leaves']
    plt.close()

    mat_data_1 = mat_data_1[idx1, :]
    mat_data_1 = mat_data_1[:, idx2]

    sorted_givers = np.array(_givers)[idx1]
    sorted_claimers = np.array(_claimers)[idx2]

    fig = plt.figure(figsize=(15, 15))

    g = sns.heatmap(mat_data_1, cmap="YlGnBu", square=True, xticklabels=sorted_claimers, yticklabels=sorted_givers,
                    annot_kws={'size':1},
               cbar_kws={"shrink": 0.1, "orientation":"vertical", "pad":0.01})
    plt.show()
    plt.close()



def extend_graph(selected_community, giver_2_claimer, claimer_2_giver):
    """
    Given a list of customers, deliver the one-step graph associated with them.

    Parameters: 
        selected_community: a list of GC customers.
        giver_2_claimer: the global giver-to-claimer mapping dict.
        claimer_2_giver: the global claimer-to-giver mapping dict.
    Returns:
        the giver and claimer lists associated with the one-step graph.
    """

    _givers = []
    _claimers = []
    for _usrid in selected_community:
        _givers.extend(claimer_2_giver[_usrid])
        _claimers.extend(giver_2_claimer[_usrid])
    for _usrid in selected_community:
        if _usrid < 974002:
            _givers.append(_usrid)
        elif _usrid > 1030347:
            _claimers.append(_usrid)
        else:
            _givers.append(_usrid)
            _claimers.append(_usrid)
    return list(set(_givers)), list(set(_claimers))

 
def filter_out_pts(X, tao_thrld):
    """
    filter out the points nearby the origin by the a fixed threshold for each dimension: keep if more than this threshold, else throw it away.

    Parameters: 
        X: the SVD spectral matrix.
        tao_thrld: the threshold for judging how much the point is close to the origin in the specific dimension.
    Returns:
        the points which don't lie around the origin in SVD spectral space.
    """

    num_row, num_col = X.shape
    thres_alldim = [max(abs(x) for x in X[:, i])/10 for i in range(num_col)]
#     print "&&&", thres_alldim
    remained_indices = []
    for i in range(num_row):
        if np.any([True if abs(X[i, j]) >= tao_thrld * thres_alldim[j] else False for j in range(num_col)]):
            remained_indices.append(i)
    return remained_indices


def cart2pol(x,y):
    """
    convert cartesian coordinates to polar coordinates, which is for vidualization of points filtered by the corresponding spectral dimension in a cartesian coordinate.

    Paramters: 
        x: the x-axis value in cartesian coordinate.
        y: the y-axis value in cartesian coordinate.
    Returns:
        the corresponding polar coordinate of (x, y)
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan(y/x if x else y/(x+.000001)) * 180/np.pi
    return rho, phi


def extract_pts_from_rays(X,Y,indices): 
    """
    extract points from three specific angular ranges in the polar coordinate. 

    Parameters:
        X: the x-axis value list of one specific group of points in the SVD spectral space.
        Y: the y-axis value list of one specific group of points in the SVD spectral space.
        indices: the indices of this specific group of points.
    Returns: 
        the points in the spectral ray. 
    """

    extracted_indices = []
    bucket0, bucket1, bucket2 = [], [], []
    for j in indices:
        _, theta = cart2pol(X[j], Y[j])
        if theta >= 75:
            bucket0.append(j)
        if theta <= -75:
            bucket1.append(j)
        if -15<= theta <=15:
            bucket2.append(j)
    extracted_indices.extend(bucket0)
    extracted_indices.extend(bucket1)
    extracted_indices.extend(bucket2)
    return extracted_indices, bucket0, bucket1, bucket2


def SpectralRay(K, adj_mat, thrld):

    """
    Take on the procedure of SpectralRayFilter.

    Parameters: 
        K: the desired dimensionality of output data
        adj_mat: the adjecency matrix feeded into SVD
        thrld: the restricted threshold used to filter the points close to the origin.
    Returns: 
        the outliers released by spectralRayFilter.
    """
    
    # singular value decomposition
    # print "SVD is going."
    U, Sigma, V = svd_new.svd_scipy(adj_mat, K)
    VT = np.transpose(V)
    # print "SVD is over."
    
    # print "TRIPWIRE is going."
    extracted_givers, extracted_claimers = defaultdict(list), defaultdict(list)
    pairs = [(u0, u1) for u0, u1 in combinations(range(K), 2)]
    pair_index = 0
    for u0, u1 in pairs:
        remained_givers = filter_out_pts(np.array(zip(U[:,u0], U[:,u1])), thrld)
        remained_claimers = filter_out_pts(np.array(zip(VT[:,u0], VT[:,u1])), thrld)
        extract_U_indices, u_bk0, u_bk1, u_bk2 = extract_pts_from_rays(U[:,u0], U[:,u1], remained_givers)
        extract_VT_indices, vt_bk0, vt_bk1, vt_bk2 = extract_pts_from_rays(VT[:,u0],VT[:,u1], remained_claimers)
        extracted_givers[pair_index] = extract_U_indices
        extracted_claimers[pair_index] = extract_VT_indices
        pair_index += 1

    # print "TRIPWIRE is over."
    
    nodesTripWire = []
    for usr in extracted_givers.values(): 
        nodesTripWire.extend(usr)
    for usr in extracted_claimers.values():
        nodesTripWire.extend(usr)
    nodesTripWire = list(set(nodesTripWire))
    
    return nodesTripWire


def spectral_2D_visual(u0, u1, U, VT, _rem_gids, _rem_cids, _extra_U, _extra_V):

    """
    Visualization in the 2-D space constructed by U components or V components.

    Parameters: 
        u0: the index of the first dimension of U components or V components.
        u1: the index of the second dimension of U components or V components.
        U: the spectral U-component matrix for all of customers
        VT: the spectral V-component transpose matrix for all of customers.
        _rem_gids: the indices of giver customers excluding the ones nearby the origin
        _rem_cids: the indices of claimer customers excluding the ones nearby the origin.
        _extra_U: the indices of givers finally lying in the ray.
        _extra_V: the indices of claimers eventually lying in the ray.
    Returns: 
       Output the 2-D visualization of the 3-group points above. 
    """

    fig = plt.figure(figsize=(40, 60))
    fig0 = fig.add_subplot(3, 2, 1)
    fig0.plot(U[:, u0], U[:, u1], 'bo', markersize=5)
    fig0.set_xlabel('U'+repr(u0))
    fig0.set_ylabel('U'+repr(u1))
    fig0.set_title('U%d against U%d' % (u0, u1))

    fig1 = fig.add_subplot(3, 2, 2)
    fig1.plot(VT[:, u0], VT[:, u1], 'ro', markersize=5)
    fig1.set_xlabel('V'+repr(u0))
    fig1.set_ylabel('V'+repr(u1))
    fig1.set_title('V%d against V%d' % (u0, u1))

    fig0 = fig.add_subplot(3, 2, 3)
    fig0.plot(U[_rem_gids, u0], U[_rem_gids, u1], 'bo', markersize=5)
    fig0.set_xlabel('U'+repr(u0))
    fig0.set_ylabel('U'+repr(u1))
    fig0.set_title('U%d against U%d' % (u0, u1))

    fig1 = fig.add_subplot(3, 2, 4)
    fig1.plot(VT[_rem_cids, u0], VT[_rem_cids, u1], 'ro', markersize=5)
    fig1.set_xlabel('V'+repr(u0))
    fig1.set_ylabel('V'+repr(u1))
    fig1.set_title('V%d against V%d' % (u0, u1))


    fig0 = fig.add_subplot(3, 2, 5)
    fig0.plot(U[_extra_U, u0], U[_extra_U, u1], 'bo', markersize=5)
    fig0.set_xlabel('U'+repr(u0))
    fig0.set_ylabel('U'+repr(u1))
    fig0.set_title('U%d against U%d' % (u0, u1))

    fig1 = fig.add_subplot(3, 2, 6)
    fig1.plot(VT[_extra_V, u0], VT[_extra_V, u1], 'ro', markersize=5)
    fig1.set_xlabel('V'+repr(u0))
    fig1.set_ylabel('V'+repr(u1))
    fig1.set_title('V%d against V%d' % (u0, u1))

    plt.show()
    
def accuracy_evaluation(sus_customers, sus_dict):
    """
    given the suspicous customers released by models and the ground-truth manually investigated, 
    measure the models' performance by the accuracy, the percent of real misuse customers out of
    suspicious customers detected by models.

    Parameters: 
        sus_customers: a list of suspicious GC customers.
        sus_dict: the ground truth list obtained by manual investigation.
    Returns:
        the percent of real misuse customers out of the total customers picked out.
    """

    _count = 0
    for _sus_customer in sus_customers:
        _count += sus_dict[_sus_customer]
    return np.round(_count/float(len(sus_customers)), decimals=4) * 100