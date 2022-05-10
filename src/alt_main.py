import numpy as np
from graph_parsing import *
from time import sleep
import networkx as nx
import numpy_indexed as npi
from argparse import Namespace
from flaml import AutoML
from sklearn.metrics import mean_absolute_error as mae
import pickle
import sys
from os.path import exists


def sigmoid_adj(x,a):
    return 1/(1 + np.exp(-1*(x-a)))

def non_joint_embedding(graph,features, treatment, outcomes, sample, sample_translation, training_nodes, \
    testing_nodes, seed):
    
    to_save = True
    automl_file = "output/non_joint_" + str(seed) + "_automl.pkl"
    embed_file = "output/non_joint_" + str(seed) + "_cur_embed.csv"
    sample_treatments_file = "output/non_joint_" + str(seed) + "_s_t.txt"
    sample_outcomes_file = "output/non_joint_" + str(seed) + "_s_o.txt"

    
    total_epochs = 16
    sample_treatments = []
    sample_outcomes = []
    for sample_vtx in sample['vertex_index']:
        orig_vtx = sample_translation[sample_vtx]
        sample_treatments.append(treatment[orig_vtx])
        sample_outcomes.append(outcomes[orig_vtx])
    sample_treatments = np.array(sample_treatments)
    np.savetxt(sample_treatments_file, sample_treatments)
    sample_outcomes = np.array(sample_outcomes)
    np.savetxt(sample_outcomes_file, sample_outcomes)

    
    my_args = {'edge_path':'unused','features_path':'unused','output_path':embed_file,'epoch_file':'unused',
                'node_embedding_dimensions':8,'feature_embedding_dimensions':8,'batch_size':64,
                'alpha':1.0,'epochs':16,'negative_samples': 15}
    arg_ns = Namespace(**my_args)

    from asne_pkg.main import run_asne
    run_asne(arg_ns,graph,features)
    cur_embed = pd.read_csv(embed_file)
    cur_loss = run_embedded_training(sample_treatments, cur_embed, sample_outcomes, training_nodes, \
        testing_nodes, to_save, automl_file)
    sys.stdout.flush()
    print('MAE Loss: ' + str(cur_loss))
    sys.stdout.flush()

    return automl_file, embed_file, sample_treatments, sample_outcomes

def prediction(sample_treatments, model_f, embedding_f,predict_outfile):
    print(sample_treatments)
    print(model_f)
    print(embedding_f)
    print(predict_outfile)
    X_ = pd.read_csv(embedding_f)
    X = X_.drop(["id"],axis=1)
    X["treatments"] = sample_treatments

    with open(model_f, "rb") as f:
        automl = pickle.load(f)

    X = X.to_numpy()
    
    outcomes = automl.predict(X)
    estimator = np.average(outcomes)
    print(estimator)

    if(not exists(predict_outfile)):
        with open(predict_outfile, 'w+') as fp:
            pass
    current_est_list = np.loadtxt(predict_outfile)
    current_est_list = np.append(current_est_list,estimator)
    np.savetxt(predict_outfile,current_est_list)




def run_embedded_training(treatments, embedding, outcomes, training_nodes, testing_nodes, to_save, outfile):
    outcomes = outcomes.reshape(-1,1)
    data = embedding.drop(["id"],axis=1)
    data["treatment"] = treatments
    X_train = data.loc[training_nodes] 
    y_train = outcomes[training_nodes]
    X_test = data.loc[testing_nodes]
    y_test = outcomes[testing_nodes]

    print('X_train')
    print(X_train)
    print('y_train')
    print(y_train)

    automl = AutoML()
    automl_settings = {
            "metric": 'mae',
            "estimator_list": 'auto',
            "task": 'classification',
            "time_budget": 250,
            "log_file_name": "./nj_automl_factual.log",
            }
    automl.fit(X_train=X_train, y_train=y_train,**automl_settings)
    y_pred = automl.predict(X_test)
    if(to_save):
        with open(outfile, "wb") as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    return mae(y_test,y_pred)
    
   

def gen_embed_train(sample,sample_translation,profiles, covariates):
    age_cat = []
    if 'scaled_age' in covariates:
        age_cat = np.where(profiles['scaled_age'] < 0., -1., 1.)
        age_cat[np.isnan(profiles['scaled_age'])] = 0
        age_cat[age_cat == -1] = 1

    raw_sample_graph = nx.from_edgelist(sample['edge_list'])
    cov_data = {}
    for sample_vtx in sample['vertex_index']:
        orig_vtx = sample_translation[sample_vtx]
        cur_feat_list = []
        for cat in covariates:
            if(cat == 'scaled_age'):
                cur_feat_list.append(int(age_cat[orig_vtx]))
            else:
                cur_feat_list.append(int(profiles[cat].iloc[[orig_vtx]]))
        cov_data[sample_vtx] = cur_feat_list

    return raw_sample_graph, cov_data

def asne_wrapper(graph,features, treatment, outcomes, sample, sample_translation, training_nodes, \
    testing_nodes, seed):
    

    epoch_file = str(seed) + "_epoch_info.txt"
    automl_file = "output/" + str(seed) + "_automl.pkl"
    embed_file = "output/" + str(seed) + "_cur_embed.csv"
    sample_treatments_file = "output/" + str(seed) + "_s_t.txt"
    sample_outcomes_file = "output/" + str(seed) + "_s_o.txt"

    
    total_epochs = 15
    sample_treatments = []
    sample_outcomes = []
    for sample_vtx in sample['vertex_index']:
        orig_vtx = sample_translation[sample_vtx]
        sample_treatments.append(treatment[orig_vtx])
        sample_outcomes.append(outcomes[orig_vtx])
    sample_treatments = np.array(sample_treatments)
    np.savetxt(sample_treatments_file, sample_treatments)
    sample_outcomes = np.array(sample_outcomes)
    np.savetxt(sample_outcomes_file, sample_outcomes)

    to_save = False
    np.savetxt(epoch_file,np.array([0.0]))
    for i in range(total_epochs):
        sys.stdout.flush()
        print('Current Epoch: ' + str(i))
        sys.stdout.flush()
        my_args = {'edge_path':'unused','features_path':'unused','output_path':embed_file,'epoch_file':epoch_file,
                    'node_embedding_dimensions':8,'feature_embedding_dimensions':8,'batch_size':64,
                    'alpha':1.0,'epochs':i,'negative_samples': 15}
        arg_ns = Namespace(**my_args)

        from asne_pkg.main import run_asne
        run_asne(arg_ns,graph,features)
        cur_embed = pd.read_csv(embed_file)
        if(i == total_epochs - 1):
            to_save = True
        cur_loss = run_embedded_training(sample_treatments, cur_embed, sample_outcomes, training_nodes, \
            testing_nodes, to_save, automl_file)
        cur_losses = np.loadtxt(epoch_file)
        cur_losses = np.append(cur_losses,cur_loss)
        sys.stdout.flush()
        print('Current MAE Loss: ' + str(cur_losses))
        sys.stdout.flush()
        np.savetxt(epoch_file,cur_losses)
    return automl_file, embed_file, sample_treatments, sample_outcomes


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

def assign_treatments(profiles,covariates, cov_coeff, seed):
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
    predict_only = True
    final_res_only = False
    non_joint = False
    #seed = 87
    #seed = 42
    #seed = 103
    #seed = 65
    seed = 233
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

    if(predict_only):
        if(not non_joint):
            model_f= "output/" + str(seed) + "_automl.pkl"
            embedding_f = "output/" + str(seed) + "_cur_embed.csv"
            sample_treatments_file = "output/" + str(seed) + "_s_t.txt"
            sample_treatments = np.loadtxt(sample_treatments_file)
            prediction(sample_treatments,model_f,embedding_f,"output/predicted_estimators.txt")
        else:
            model_f= "output/non_joint_" + str(seed) + "_automl.pkl"
            embedding_f = "output/non_joint_" + str(seed) + "_cur_embed.csv"
            sample_treatments_file = "output/non_joint_" + str(seed) + "_s_t.txt"
            sample_treatments = np.loadtxt(sample_treatments_file)
            prediction(sample_treatments,model_f,embedding_f,"output/non_joint_predicted_estimators.txt")
        return


    categories = ['I_like_books','I_like_movies','I_like_music','relation_to_smoking','scaled_age']
    cat_coeff = gen_covar_coeff(categories, seed)

    if(init_processing):
        one_time_graph_processing()
    
    graph, profiles = graph_processing(True)
    profiles = second_profile_process(profiles)
    print('Finished graph input processing')

    treatments = assign_treatments(profiles,categories, cat_coeff, seed)
    outcomes = assign_outcomes(treatments, graph, profiles, categories, cat_coeff, seed)
    temp_writes(treatments,outcomes)
    print('Finished treatment and outcome simulated assignment')

    sample, sample_translation = sample_graph(graph,seed)
    print('Finished sampling')
    training_nodes, testing_nodes = train_test_node_split(sample,seed)
    print('Finished train-test split')

    graph_embed, graph_features = gen_embed_train(sample,sample_translation,profiles, categories)
    print('Finished generating training set')

    if(non_joint):
        model_f, embedding_f, sample_treatments, sample_outcomes = non_joint_embedding(graph_embed, graph_features,treatments, \
        outcomes,sample,sample_translation, training_nodes, testing_nodes, seed)
        print('Finished training')

        prediction(sample_treatments,model_f,embedding_f,"output/non_joint_predicted_estimators.txt")
        print('Finished predictions on sample')
    else:
        model_f, embedding_f, sample_treatments, sample_outcomes = asne_wrapper(graph_embed, graph_features,treatments, \
            outcomes,sample,sample_translation, training_nodes, testing_nodes, seed)
        print('Finished training')

        prediction(sample_treatments,model_f,embedding_f,"output/predicted_estimators.txt")
        print('Finished predictions on sample')


if __name__ == '__main__':
    main()
