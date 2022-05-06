import numpy as np
from graph_parsing import *
from time import sleep

def sim():
    pass

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
    graph = {"edge_list":edge_list,"weights":weights,"adjacency_list":adjacency_list,"num_vertices":len(vertex_index)}

    return graph, profiles


def main():
    one_time_graph_processing()
    graph, profiles = graph_processing(True)


if __name__ == '__main__':
    main()