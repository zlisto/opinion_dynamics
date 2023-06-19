from scipy import integrate
from scipy.integrate import odeint
from scipy.sparse import coo_matrix,diags


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_us_election():
    df = pd.read_csv("data/2016_Second_Presidential_Debate_full.csv")
    ndays_data = 312
    rates = np.array(df.rate)/ndays_data  #make rates in tweets/day
    opinions0 = np.array(df.opinion_tweet)

    edge_list = []
    vdict = {}

    for index, row in df.iterrows():    
        v = str(row.user_id)
        vdict[v] = index
        if type(row.friend_names) == str:
            friends = row.friend_names.split(",")
            for friend in friends:
                edge = (friend,v)
                edge_list.append(edge)
    rows, cols = [], []
    rows_E = []
    ne = len(edge_list)
    nv = len(vdict.keys())
    data = np.ones(ne)
    for edge in edge_list:
        u,v = edge[0], edge[1]
        rows.append(vdict[v])  #follower
        cols.append(vdict[u])  #friend/following
        rows_E.append(vdict[v])

    cols_E = np.arange(0,ne)

    A = coo_matrix((data, (rows, cols)), shape=(nv, nv))
    E = coo_matrix((data, (rows_E, cols_E)), shape=(nv, ne))
    network_params = {'A':A, 'E':E, 'rates':rates, 'opinions0':opinions0}
    return network_params

def load_giletsjaunes():
    df = pd.read_csv("data/GiletsJaunes_user_polarities_final.csv", sep = ";")
    ndays = 119
    rates = np.array(df.n_tweets)/ndays  #make rates in tweets/day
    opinions0 = np.array(df.user_polarity)
    assert len(rates)==len(opinions0)
    vdict = {}

    for index, row in df.iterrows():    
        v = str(row.id.astype(int))
        vdict[str(row.id).replace('.0',"")] = index
    nv = len(vdict.keys())    

    with open('data/GiletsJaunes_full_graph.csv', 'r') as file:
        # Read each line and store it in the list
        edge_list = []
        for line in file:
            x = line.split(';')
            v = x[0]
            friends = x[1:]
            for friend in friends:
                edge = (friend,v)
                edge_list.append(edge)
        rows, cols = [], []
        rows_E = []
        ne = 0

        for edge in edge_list:
            u,v = edge[0], edge[1]
            if u in vdict.keys() and v in vdict.keys():
                rows.append(vdict[v])  #follower
                cols.append(vdict[u])  #friend/following
                rows_E.append(vdict[v])
                ne+=1

        cols_E = np.arange(0,ne)
        data = np.ones(ne)

        A = coo_matrix((data, (rows, cols)), shape=(nv, nv))
        E = coo_matrix((data, (rows_E, cols_E)), shape=(nv, ne))
        network_params = {'A':A, 'E':E, 'rates':rates, 'opinions0':opinions0} 
        return network_params

def load_brexit_sample():
  path = 'data/'
  fname_opinion_rate = f"{path}Brexit_sample_01.csv"
  fname_adjlist  = f"{path}Brexit_sample.adjlist"

  df = pd.read_csv(fname_opinion_rate)
  G = nx.read_adjlist(fname_adjlist)
  V = [v for v in G.nodes()]
  df = df[df.user_id.astype(str).isin(V)]
  opinions_initial = df.opinion_tweet.array
  rate = df.rate.array

  mapping = {}
  mapping_rev = {}
  c=0
  for index, row in df.iterrows():
    c+=1
    mapping[str(row.user_id)] = c
    mapping_rev[c] = str(row.user_id)
  H = nx.relabel_nodes(G, mapping)
  return H, np.array(opinions_initial), np.array(rate) , mapping_rev

def load_giletsjaunes_sample():
  path = 'data/'
  fname_opinion_rate = f"{path}GiletsJaunes_sample_02.csv"
  fname_adjlist  = f"{path}GiletsJaunes_sample_02.adjlist"

  df = pd.read_csv(fname_opinion_rate)
  df['user_id'] = df.user_id.astype(str)

  G = nx.read_adjlist(fname_adjlist)
  V = [v for v in G.nodes()]
  df = df[df.user_id.astype(str).isin(V)]
  opinions_initial = df.opinion_tweet.array
  rate = df.rate.array

  mapping = {}
  mapping_rev = {}
  c=0
  for index, row in df.iterrows():
    c+=1
    mapping[str(row.user_id)] = c
    mapping_rev[c] = str(row.user_id)
  H = nx.relabel_nodes(G, mapping)
  return H, np.array(opinions_initial), np.array(rate) , mapping_rev