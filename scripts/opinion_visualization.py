
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde


#########################################################
#Plotting functions
def plot_opinions(T, Opinions, U = None):
    plt.plot(T, Opinions)
    if U is not None:
        plt.plot(T, 1-U.mean(axis=1), '.-', color = 'purple', label='Mean control')
        plt.legend()
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Opinions")
    plt.ylim([0,1.1])
    
    
def plot_opinion_quantiles(T ,Opinions, q=[50, 25, 75, 5, 95], U = None):
    quantiles = np.percentile(Opinions, q=q, axis=1)

    # Plot quantiles
    plt.plot(T, quantiles[0], color='black', label='Mean')
    plt.fill_between(T, quantiles[1], quantiles[2], color='blue', alpha=0.5, label='25th-75th Quantiles')
    plt.fill_between(T, quantiles[3], quantiles[4], color='pink', alpha=0.5, label='5th-95th Quantiles')
    if U is not None:
        plt.plot(T, 1-U.mean(axis=1), '.-', color = 'purple', label='Mean control')
        plt.legend()
    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Opinions')
    #plt.legend(loc = 4)
    plt.grid()
    

def draw_network(G):
    nv = G.number_of_nodes()    
    if nv<=100:
        colors = []
        for v in G.nodes():
            #print(f"{v}: {G.nodes[v]}")
            if (G.nodes[v]['opinion']<0.5) & (G.nodes[v]['opinion']>=0):
                colors.append('blue')
            elif (G.nodes[v]['opinion']>0.5) & (G.nodes[v]['opinion']<=1):
                colors.append('red')
            elif G.nodes[v]['opinion']==0.5:
                colors.append('purple')
            else:
                colors.append('green')
        pos = nx.kamada_kawai_layout(G)
        fig = plt.figure(figsize = (3,3))
        nx.draw(G,pos, node_color = colors, node_size = 50)
        plt.show()
    else:
        print("Network has more than 100 nodes.  Dont draw it cuz it takes too long")
        fig = None
    return fig
    


