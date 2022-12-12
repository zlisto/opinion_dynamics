from scipy import integrate
from scipy.integrate import odeint
from scipy.sparse import coo_matrix,diags


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_brexit():
  path = '/content/drive/MyDrive/SocialNetworkAndOpinionDatasets/Brexit/'
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
  path = '/content/drive/MyDrive/SocialNetworkAndOpinionDatasets/GiletsJaunes/'
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