
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde



def plot_opinions(opinions, agent_opinions, dt, reward = 0, quantile_plot = False):
  nv = opinions.shape[1]  #number of nodes
  assert opinions.shape[0] == len(agent_opinions)
  nsteps = opinions.shape[0]
  T = np.arange(0, nsteps*dt, dt)
  fig = plt.subplot(1,1,1)
  if quantile_plot == True:
    print("quantile plot")
    Q05 = np.quantile(opinions,0.05,axis = 1)
    Q95 = np.quantile(opinions,0.95,axis = 1)
    Q50 = np.quantile(opinions,0.50,axis = 1)
    Q25 = np.quantile(opinions,0.25,axis = 1)
    Q75 = np.quantile(opinions,0.75,axis = 1)

    plt.plot(T,Q05,color = 'skyblue', linewidth = 1)
    plt.plot(T,Q95,color = 'skyblue', linewidth = 1)
    plt.fill_between(T, Q05, Q95,  color='skyblue', alpha=0.5)

    plt.plot(T,Q25,color = 'blue', linewidth = 1)
    plt.plot(T,Q75,color = 'blue', linewidth = 1)
    plt.fill_between(T, Q25, Q75, color='blue', alpha=0.4)
    plt.plot(T,Q50,color = 'darkblue', linewidth = 2)
  else:
    for k in range(nv):
      plt.plot(T,opinions, label = f"{k}")
  
  plt.ylabel('Opinion')
  plt.xlabel('Time')
  plt.plot(T,agent_opinions, linewidth = 3, 
          color = 'red', label = 'Agent')
  plt.title(f"R(T) = {reward:.4f}")
  plt.legend()
 
  return fig
# <<<<<<< HEAD



def plot_opinion_quantiles(T:np.ndarray,Opinions:np.ndarray):
    quantiles = np.percentile(Opinions, q=[50, 25, 75, 5, 95], axis=1)

    # Plot quantiles
    plt.plot(T, quantiles[0], color='black', label='Mean')
    plt.fill_between(T, quantiles[1], quantiles[2], color='blue', alpha=0.5, label='25th-75th Quantiles')
    plt.fill_between(T, quantiles[3], quantiles[4], color='pink', alpha=0.5, label='5th-95th Quantiles')

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Opinions')
    #plt.legend(loc = 4)
    plt.grid()
    
def plot_opinion_heatmap(T,Opinions, cmap = 'jet'):
    yopinions = np.linspace(0,1,100)
    Z = []
    XT = []
    Yopinions = []
    for tind in range(Opinions.shape[0]):
        kde = gaussian_kde(Opinions[tind,:])
        Z.append(kde(yopinions))
        XT.append(T[tind]*np.ones(len(T)))
        Yopinions.append(yopinions)

    plt.pcolormesh(XT, Yopinions, Z, cmap= cmap)
    plt.colorbar()
    return XT,Yopinions,Z   

def plot_opinion_adjoint(T, Opinions, P, agents_opinions=None):
    plt.subplot(1,2,1)
    plt.plot(T, Opinions)
    plt.ylabel('Opinion',fontsize = 14)
    plt.xlabel('Time',fontsize = 14)
    plt.ylim([0,1.1])
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    plt.grid()
    if agents_opinions is not None:
        plt.plot(T, agents_opinions, color = 'red', label = 'agent', marker=".")
        plt.legend()



    plt.subplot(1,2,2)
    plt.plot(T,P)
    plt.ylabel('Adjoint',fontsize = 14)
    plt.xlabel('Time',fontsize = 14)
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    plt.grid()
    

def visualize_network(G0:nx.DiGraph,Opinions0:np.ndarray, Rates:np.ndarray):
    # Visualize network if ain't too big
    G = None
    if G0.number_of_nodes()<=100:
        G = nx.DiGraph()
        for cnt,v in enumerate(G0.nodes()):
            opinion = Opinions0[cnt]
            rate = Rates[cnt]
            G.add_node(v, Name =v, InitialOpinion=opinion, Rate=1)
            followers = G0.successors(v)
        for follower in followers:
            print(f"edge {(v,follower)}")
            G.add_edge(v,follower,weight = rate)

        cnt = 0
        for targets,agent_rate in zip(Targets,agent_rates):
            G.add_node(f"A{cnt}")
        for v in targets:
            G.add_edge(f"A{cnt}",v, weight = agent_rate)
            cnt+=1
            edgewidth = [ 1*d['weight'] for (u,v,d) in G.edges(data=True)]

        pos = nx.kamada_kawai_layout(G)  # positions for all nodes
        pos["Agent"] = (0,-2)
        fig = plt.figure(figsize=(8,8))
        nx.draw(G, pos,width=1.5,node_color="red",edge_color="purple",node_size=500)
        nx.draw_networkx_edges(G, pos, width=edgewidth,edge_color="purple")
        nx.draw_networkx_labels(G, pos,font_size=17,font_color="black")
        fig.set_facecolor("#00000F")
        plt.show()
    else:
        print(f"network has {G0.number_of_nodes()} nodes, too big to draw")
    return G


# =======
# >>>>>>> 07f59b514762b3daf99f9d8afcc0ec41118c5126
