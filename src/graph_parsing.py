import os
import argparse
from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf
from time import sleep
import time 


from relational_erm.graph_ops.representations import PackedAdjacencyList
from relational_erm.graph_ops.representations import create_packed_adjacency_from_redundant_edge_list

def load_profiles(profile_file):

    names = str.split("user_id public completion_percentage gender region last_login registration age body "
                      "I_am_working_in_field spoken_languages hobbies I_most_enjoy_good_food pets body_type "
                      "my_eyesight eye_color hair_color hair_type completed_level_of_education favourite_color "
                      "relation_to_smoking relation_to_alcohol sign_in_zodiac on_pokec_i_am_looking_for love_is_for_me "
                      "relation_to_casual_sex my_partner_should_be marital_status children relation_to_children "
                      "I_like_movies I_like_watching_movie I_like_music I_mostly_like_listening_to_music "
                      "the_idea_of_good_evening I_like_specialties_from_kitchen fun I_am_going_to_concerts "
                      "my_active_sports my_passive_sports profession I_like_books life_style music cars politics "
                      "relationships art_culture hobbies_interests science_technologies computers_internet education "
                      "sport movies travelling health companies_brands more")
    usecols = str.split("user_id public completion_percentage gender region last_login registration age "
                        "completed_level_of_education sign_in_zodiac relation_to_casual_sex I_like_books ")


    profiles = pd.read_csv(profile_file, names=names, index_col=False, usecols=usecols, header=None, sep='\t')
    profiles.set_index('user_id', inplace=True, drop=False)

    profiles['region'] = profiles['region'].astype('category')
    profiles['public'] = profiles['public'].astype('category')
    profiles['gender'] = profiles['gender'].astype('category')
    profiles.loc[profiles.age == 0, 'age'] = np.nan

    profiles['completed_level_of_education'] = profiles['completed_level_of_education'].isna()
    profiles['sign_in_zodiac'] = profiles['sign_in_zodiac'].isna()
    profiles['relation_to_casual_sex'] = profiles['relation_to_casual_sex'].isna()
    profiles['I_like_books'] = profiles['I_like_books'].isna()

    profiles['last_login'] = pd.to_datetime(profiles['last_login'])
    profiles['registration'] = pd.to_datetime(profiles['registration'])
    profiles['user_id'] = profiles['user_id'] - 1

    return profiles


def preprocess_data():
    link_file = '../data/soc-pokec-relationships.txt'
    profile_file = '../data/soc-pokec-profiles.txt'

    print('Loading Relationship File')
    start = time.time()
    edge_list = np.loadtxt(link_file, dtype=np.int32)
    edge_list = edge_list - 1
    end = time.time()
    print('Completed Relationship Loading, Time taken was: ' + str(end - start))

    print('Loading Profiles File')
    start = time.time()
    profiles = load_profiles(profile_file)
    end = time.time()
    print('Completed Profiles Loading, Time taken was: ' + str(end-start))
    
    return edge_list, profiles


def preprocess_packed_adjacency_list(edge_list):

    orig_edge_list = edge_list.copy()
    edge_list = edge_list[edge_list[:, 0] != edge_list[:, 1], :]
    edge_list.sort(axis=-1)
    edge_list = np.unique(edge_list, axis=0)

    edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))
    adjacency_list = create_packed_adjacency_from_redundant_edge_list(edge_list)

    return {
        'neighbours': adjacency_list.neighbours,
        'offsets': adjacency_list.offsets,
        'lengths': adjacency_list.lengths,
        'vertex_index': adjacency_list.vertex_index,
        'edge_list': orig_edge_list
    }


def _edges_in_region(edge_list, vertices_in_region):
    edge_list = np.copy(edge_list)
    edge_in_region = np.isin(edge_list[:, 0], vertices_in_region)
    edge_list = edge_list[edge_in_region]
    edge_in_region = np.isin(edge_list[:, 1], vertices_in_region)
    edge_list = edge_list[edge_in_region]
    return edge_list.shape[0]


def subset_to_region(edge_list, profiles, regions=None):

    if regions is None:
        #
        regions = ['zilinsky kraj, zilina', 'zilinsky kraj, cadca', 'zilinsky kraj, namestovo']

    user_in_region = np.zeros_like(profiles['region'], dtype=np.bool)
    for region in regions:
        user_in_region = np.logical_or(user_in_region,
                                       profiles['region'] == region)

    vertices_in_region = profiles.loc[user_in_region]['user_id'].values

    edge_list = np.copy(edge_list)
    edge_in_region = np.isin(edge_list[:, 0], vertices_in_region)
    edge_list = edge_list[edge_in_region]
    edge_in_region = np.isin(edge_list[:, 1], vertices_in_region)
    edge_list = edge_list[edge_in_region]


    present_user_ids = np.unique(edge_list)
    present_user_indicator = np.isin(profiles['user_id'].values, present_user_ids)
    regional_profiles = profiles[present_user_indicator]

    regional_profiles.set_index('user_id')

    index_to_user_id = regional_profiles['user_id'].values
    user_id_to_index = np.zeros(np.max(index_to_user_id)+1, dtype=np.int32)-1
    user_id_to_index[index_to_user_id] = np.arange(index_to_user_id.shape[0])

    edge_list = user_id_to_index[edge_list]

    regional_profiles.to_pickle("../data/regional_profiles.pkl")
    #np.savez_compressed('regional_links.npz', edge_list=edge_list)
    packed_adjacency_list_data = preprocess_packed_adjacency_list(edge_list)
    np.savez_compressed('../data/regional_links.npz', **packed_adjacency_list_data)


def process_pokec_attributes(profiles):

    included_features = str.split("public "
                                  "completion_percentage "
                                  "gender "
                                  "region "
                                  "age "
                                  "completed_level_of_education "
                                  "sign_in_zodiac "
                                  "relation_to_casual_sex "
                                  "I_like_books "
                                  "recent_login "
                                  "old_school "
                                  "scaled_registration "
                                  "scaled_age "
                                  )

    profiles['recent_login'] = (profiles['last_login'] < pd.Timestamp(2012, 5, 1))
    profiles['old_school'] = (profiles['registration'] < pd.Timestamp(2009, 1, 1))

    profiles['scaled_registration'] = (profiles['registration'] - profiles['registration'].min()) / pd.offsets.Day(1)
    profiles['scaled_registration'] = (profiles['scaled_registration'] - profiles['scaled_registration'].mean())/(profiles['scaled_registration'].std())

    profiles['scaled_age'] = (profiles['age'] - profiles['age'].mean())/(profiles['age'].std())


    cat_columns = profiles.select_dtypes(['category']).columns
    profiles[cat_columns] = profiles[cat_columns].apply(lambda x: x.cat.codes)
    profiles[cat_columns] = profiles[cat_columns].apply(lambda x: x.astype(np.int32))
    bool_columns = profiles.select_dtypes([bool]).columns
    profiles[bool_columns] = profiles[bool_columns].apply(lambda x: x.astype(np.int32))

    profiles['age'][profiles['age'].isna()] = -1.

    profiles['age'] = profiles['age'].astype(np.float32)
    profiles['completion_percentage'] = profiles['completion_percentage'].astype(np.float32)

    pokec_features = {}
    for feature in included_features:
        pokec_features[feature] = profiles[feature].values

    return pokec_features
