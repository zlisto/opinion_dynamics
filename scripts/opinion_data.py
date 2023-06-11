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

    rates = np.array(df.rate)
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



def load_brexit():
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

def load_giletsjaunes():
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