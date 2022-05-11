import numpy as np
from graph_parsing import *
from time import sleep
import networkx as nx
import numpy_indexed as npi
from argparse import Namespace
from sklearn.metrics import mean_absolute_error as mae, confusion_matrix
import pickle
from scipy.io import mmwrite, mmread
import sys
from os.path import exists
from scipy import sparse
from paican_pkg.paican import PAICAN


def sigmoid_adj(x,a):
    return 1/(1 + np.exp(-1*(x-a)))

def community_generation(graph, covariates, profiles):
    comm_ct = 100
    age_cat = []
    id_arr = np.array(profiles['user_id'])
    print(graph['vertex_index'])
    if 'scaled_age' in covariates:
        age_cat = np.where(profiles['scaled_age'] < 0., -1., 1.)
        age_cat[np.isnan(profiles['scaled_age'])] = 0
        age_cat[age_cat == -1] = 1

    adj_mat = np.zeros((len(graph['vertex_index']),len(graph['vertex_index'])))
    attrs = np.zeros((len(graph['vertex_index']),len(covariates)))

    print('Writing attribute matrix')
    for i in range(len(graph['vertex_index'])):
        if(i % 100 == 0):
            print('Node ' + str(i) + ' completed.')
        cur_neighbors = graph['neighbours'][graph['offsets'][i]:graph['offsets'][i]+graph['lengths'][i]]
        adj_mat[i][cur_neighbors] = 1

        for k in range(len(covariates)):
            if(covariates[k] == 'scaled_age'):
                attrs[i][k] = age_cat[i]
            else:
                attrs[i][k] = profiles[covariates[k]][id_arr[i]]

    A = sparse.csr_matrix(adj_mat)
    X = sparse.csr_matrix(attrs)
    
    paican = PAICAN(A, X, comm_ct, verbose=True)
    z_pr, ca_pr, cx_pr = paican.fit_predict()
    np.savetxt('z_pr_tmp.txt',z_pr)
    np.savetxt('ca_pr_tmp.txt',ca_pr)
    np.savetxt('cx_pr_tmp.txt',cx_pr)

def sample_graph(graph,seed):
    #Let's use p-sampling
    np.random.seed(seed)
    p = 0.4
    vtx_sel = np.random.binomial(1,[p] * (len(graph['vertex_index'])))
    idxs = np.where(vtx_sel == 1)
    sampled_vtxs = graph['vertex_index'][idxs]


    offsets_to_check = graph["offsets"][sampled_vtxs]
    lengths_to_check = graph["lengths"][sampled_vtxs]
    new_edge_list = []
    final_sampled_vtxs = []
    temp_neighbor_list = []
    total_neighbor_list = []
    for i in range(len(sampled_vtxs)):
        cur_neighbors = graph["neighbours"][offsets_to_check[i]:offsets_to_check[i]+lengths_to_check[i]]
        final_cur_neighbors = []
        for neighbor in cur_neighbors:
            if(neighbor in sampled_vtxs):
                final_cur_neighbors.append(neighbor)
                total_neighbor_list.append(neighbor)
        if(len(final_cur_neighbors) == 0):
            continue
        final_sampled_vtxs.append(sampled_vtxs[i])
        temp_neighbor_list.append(final_cur_neighbors)

    final_sampled_vtxs = np.array(final_sampled_vtxs)
    temp_neighbor_list = np.array(temp_neighbor_list)
    total_neighbor_list = np.array(total_neighbor_list)
    neighbor_idxs = npi.indices(final_sampled_vtxs,total_neighbor_list)
    neighbor_conversion = dict(zip(total_neighbor_list,neighbor_idxs))
    translation = dict(zip(np.arange(len(final_sampled_vtxs)), final_sampled_vtxs))

    for i in range(len(final_sampled_vtxs)):
        for n in temp_neighbor_list[i]:
            new_edge_list.append([i, neighbor_conversion[n]])

    new_edge_list = np.array(new_edge_list)
    new_adj_list = preprocess_packed_adjacency_list(new_edge_list)

    sample_graph = {'neighbours': new_adj_list['neighbours'],
        'offsets': new_adj_list['offsets'],
        'lengths': new_adj_list['lengths'],
        'vertex_index': new_adj_list['vertex_index'],
        'weights': new_adj_list['weights'],
        'edge_list': new_edge_list}
    return sample_graph, translation


def propensity_linear_function(values,cov_coeff):
    for i in range(len(values)):
        values[i] *= cov_coeff[i]
    confounding = np.sum(values,axis=0)
    propensities = sigmoid_adj(confounding,len(values)/2.0)
    return propensities
    
def gen_covar_coeff(covariates, seed):
    np.random.seed(seed)
    return np.random.uniform(size=len(covariates))

def assign_treatments(profiles,covariates, cov_coeff, seed,all_x): 
    np.random.seed(seed)
    
    age_cat = []
    if 'scaled_age' in covariates:
        age_cat = np.where(profiles['scaled_age'] < 0., -1., 1.)
        age_cat[np.isnan(profiles['scaled_age'])] = 0
        age_cat[age_cat == -1] = 1
    values = []

    for cat in covariates:
        if(cat == 'scaled_age'):
            values.append(age_cat)
        else:
            values.append(profiles[cat])
    values = np.array(values)

    propensities = propensity_linear_function(values,cov_coeff)
    treatments = np.random.binomial(1,propensities)
    if(all_x == 0):
        treatments = np.zeros(len(propensities))
    if(all_x == 1):
        treatments = np.ones(len(propensities))
    return treatments

def assign_outcomes(treatments,graph,profiles, covariates, cov_coeff, seed):
    np.random.seed(seed)
    confounding_alpha = np.random.uniform()
    age_cat = []
    if 'scaled_age' in covariates:
        age_cat = np.where(profiles['scaled_age'] < 0., -1., 1.)
        age_cat[np.isnan(profiles['scaled_age'])] = 0
        age_cat[age_cat == -1] = 1
    values = []
    for cat in covariates:
        if(cat == 'scaled_age'):
            values.append(age_cat)
        else:
            values.append(profiles[cat])
    values = np.array(values)
    confounding_propensities = propensity_linear_function(values,cov_coeff)

    V = []
    for i in range(len(treatments)):
        cur_neighbors = graph['neighbours'][graph['offsets'][i]:graph['offsets'][i]+graph['lengths'][i]]
        v_i = np.sum(treatments[cur_neighbors])/len(cur_neighbors)
        V.append(v_i)
    V = np.array(V)

    outcome_propensities = (1 - confounding_alpha)*V + confounding_alpha*confounding_propensities
    
    np.random.seed(seed)
    outcomes = np.random.binomial(1, outcome_propensities)
    return outcomes

def train_test_node_split(sample,seed):
    np.random.seed(seed)
    train_ratio = 0.7
    vtx_sel = np.random.binomial(1,[train_ratio] * (len(sample['vertex_index'])))
    idxs = np.where(vtx_sel == 1)
    train_vtxs = sample['vertex_index'][idxs]
    mask = np.ones(len(sample['vertex_index']), dtype=bool)
    mask[train_vtxs] = False
    test_vtxs = sample['vertex_index'][mask]

    return train_vtxs, test_vtxs


def one_time_graph_processing():
    edges, profiles = preprocess_data()
    profiles.to_pickle("../data/profiles.pkl")
    packed_adjacency_list_data = preprocess_packed_adjacency_list(edges)
    np.savez_compressed('../data/links_processed.npz', **packed_adjacency_list_data)
    subset_to_region(edges, profiles, regions=None)

def graph_processing(regional):
    profile_file = "../data/profiles.pkl"
    edge_file = "../data/links.npz"

    if(regional):
        profile_file = "../data/regional_profiles.pkl"
        edge_file = "../data/regional_links.npz"

    profiles = pd.read_pickle(profile_file)

    loaded = np.load(edge_file, allow_pickle=False)
    neighbours = loaded['neighbours']
    offsets = loaded['offsets']
    edge_list = loaded['edge_list']
    lengths = loaded['lengths']
    vertex_index = loaded['vertex_index']
    weights = np.ones(edge_list.shape[0], dtype=np.float32)
    adjacency_list = PackedAdjacencyList(neighbours, weights, offsets, lengths, vertex_index)
    graph = {'neighbours': adjacency_list.neighbours,
        'offsets': adjacency_list.offsets,
        'lengths': adjacency_list.lengths,
        'vertex_index': adjacency_list.vertex_index,
        'weights': adjacency_list.weights,
        'edge_list': edge_list}

    return graph, profiles

def final_result(predict_outfile):
    estimators = np.loadtxt(predict_outfile)
    m = np.mean(estimators)
    ste = np.std(estimators)/np.sqrt(len(estimators))
    print("Peer Influence Estimator of: " + str(m) + " with standard error of " + str(ste) + ".")

def temp_writes(treatments, outcomes):
    np.savetxt("temp_treatments.txt", treatments)
    np.savetxt("temp_outcomes.txt", outcomes)

def main():

    init_processing = False
    final_res_only = False
    all_x = 0
    seed = 87
    #seed = 42
    #seed = 103
    #seed = 65
    #seed = 233
    #seed = 32
    #seed = 83
    #seed = 44
    #seed = 55
    #seed = 19

    if(final_res_only):
        if(non_joint):
            final_result("output/non_joint_predicted_estimators.txt")
        else:
            final_result("output/predicted_estimators.txt")
        return

    categories = ['I_like_books','I_like_movies','I_like_music','relation_to_smoking','scaled_age']
    cat_coeff = gen_covar_coeff(categories, seed)

    if(init_processing):
        one_time_graph_processing()
    
    graph, profiles = graph_processing(True)
    profiles = second_profile_process(profiles)
    print('Finished graph input processing')

    treatments = assign_treatments(profiles,categories, cat_coeff, seed,all_x)
    outcomes = assign_outcomes(treatments, graph, profiles, categories, cat_coeff, seed)
    temp_writes(treatments,outcomes)
    print('Finished treatment and outcome simulated assignment')

    communities = community_generation(graph, categories, profiles)
    print('Finished community detection!')

    sample, sample_translation = sample_graph(graph,seed)
    print('Finished sampling')

    


    
    training_nodes, testing_nodes = train_test_node_split(sample,seed)
    print('Finished train-test split')

    print(sample)
    print(training_nodes)

    

if __name__ == '__main__':
    main()
