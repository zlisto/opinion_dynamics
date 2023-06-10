import networkx as nx
import pandas as pd
import numpy as np

from scripts.class_dynamics import *


def df_to_G(df):
  # Make id and names string type
  df['user_id'] = df['user_id'].astype(str)
  df['friend_names'] = df['friend_names'].astype(str)
  
  # Create G_bots that includes users and bots, but bots do not follow other bots
  G_bots = nx.DiGraph()
  df_bots = pd.DataFrame({'bot_id': [], 'bot_opinion': [], 'bot_rate': [], 'friend_names': []})
  for index, row in df.iterrows():
    user_id = row["user_id"]
    stubborn = row['stubborn']
    rate = row['rate']/30 #tweets/day
    opinion_tweet = row['opinion_tweet']
    friend_names = row["friend_names"]

    if row["bot"]==1:
      new_row = {'bot_id': user_id, 'bot_opinion': opinion_tweet, 'bot_rate': rate}
      df_bots.loc[len(df_bots)] = new_row
    else:#user not bot
      followings = friend_names.split(",")
      flag_following_valid = False
      for following in followings:
        if following in df['user_id'].values:#if following is a valid user (in user_id)
          G_bots.add_edge(following,user_id)
          flag_following_valid = True
      if not flag_following_valid:#not following any valid usuer
        G_bots.add_node(user_id)

      attributes = {'stubborn': stubborn, 'rate': rate, 'opinion': opinion_tweet}
      nx.set_node_attributes(G_bots, {user_id: attributes})
  
  G = G_bots.copy()

  # Remove bots from G
  bots_to_remove = list(df_bots["bot_id"])
  G.remove_nodes_from(bots_to_remove)

  # Create opinions_initial, rate, and mapping dict
  opinions_initial = []
  rate = []
  mapping = {}
  mapping_rev = {}
  cnt = 0
  for user_id, node_attrs in G.nodes(data=True):
    opinions_initial.append(node_attrs['opinion'])
    rate.append(node_attrs['rate'])
    mapping[cnt] = user_id
    mapping_rev[user_id] = cnt
    cnt += 1

  G_relab = nx.relabel_nodes(G, mapping_rev)#relabel user as integer

  # Create bots_action
  bots_action = []#list of Action objects for bots
  G_bots_relab = nx.relabel_nodes(G_bots, mapping_rev)#relabel user as integer, bot as string
  connected_bots = set(bots_to_remove).intersection(set(G_bots_relab.nodes()))
  for connected_bot in connected_bots:
    row_ref = (df['user_id'] == connected_bot)
    bot_opinion = df.loc[row_ref, 'opinion_tweet'].item()
    bot_rate = df.loc[row_ref, 'rate'].item()/30 #tweets/day
    target_indices = list(G_bots_relab.successors(connected_bot))#bot target indices
    bot_action = Action(connected_bot,bot_opinion,bot_rate,target_indices)
    bots_action.append(bot_action)

  return G_relab, np.array(opinions_initial), np.array(rate), mapping, bots_action, G_bots_relab


def G_to_params(fname):
  path_data = ""
  df_raw = pd.read_csv(path_data + fname)
  df = df_raw.copy(deep=True)

  G, opinions_initial, rate, mapping, bots_action, G_bots = df_to_G(df)

  # Extract stubborn users
  stub_indices = []
  n_stubborn = 0
  for node in G.nodes():
    if G.nodes[node]['stubborn'] == 1:
      stub_indices.append(node)
      n_stubborn += 1

#   objective = "max_mean" # "max/min_mean/variance"
#   objective_opt = objective.split("_")[0] # max or min
#   objective_function = objective.split("_")[1] # mean or variance
#   if objective_opt == "max":
#     objective_sign = 1
#   else: # "min"
#     objective_sign = -1

  nv = G.number_of_nodes()

  print(f"File name = {fname}")
  print(f"Nodes in df = {len(df)} including users, bots, agents")

  print(f"Nodes in G_bots = {G_bots.number_of_nodes()} including connected users, connected bots")
  print(f". Connected users = {nv}")
  n_bots = len(bots_action)
  print(f". Connected bots = {n_bots} = {n_bots/G_bots.number_of_nodes()*100:.0f}% of connected nodes")
  print(f". Stubborn users = {n_stubborn} = {n_stubborn/nv*100:.0f}% of connected users")
  print(f". Mean user rate = {rate.mean():.2f} tweets/day")
  bots_rate = [bot_action.rate for bot_action in bots_action]
  mean_bots_rate = np.mean(bots_rate)
  print(f". Mean bot rate = {mean_bots_rate:.2f} tweets/day")

  bots_targets = [bot_action.targets for bot_action in bots_action]
  bots_targets_set = set([item for sublist in bots_targets for item in sublist])
  bots_targets_times = sum(len(bot_targets) for bot_targets in bots_targets)
  print(f". Bots targets = {bots_targets_times} user-times = {len(bots_targets_set)} unique users = {len(bots_targets_set)/nv*100:.0f}% of connected users")

  G0 = G.copy()

  #adjacency matrix of network
  A = nx.adjacency_matrix(G0)
  A = A.tocoo()
  assert nv == A.shape[0]  #number of nodes in network should equal shape of A
    
  rates = np.array([G0.nodes[v]["rate"] for v in G0.nodes()])

  opinions0 = np.array([G0.nodes[v]["opinion"] for v in G0.nodes()])
  plt.hist(opinions0)
  plt.show()
    
  #adjacency matrix of network
  # A = nx.adjacency_matrix(G)
  # A = A.tocoo()
  E0 = nx.incidence_matrix(G,oriented=True)
  E0 = E0.tocoo()
  ind = E0.data>0
  E = coo_matrix((E0.data[ind], (E0.row[ind], E0.col[ind])), E0.shape)  #incidence matrix with only tail of edge

  network_params = {'A':A, 'rates':rates, 'opinions0':opinions0, 'E':E}

  return G0, network_params