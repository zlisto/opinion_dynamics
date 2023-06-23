import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde # calculate kernel density


#########################################################
#Plotting functions
def plot_opinions(T, Opinions, U = None):
    plt.plot(T, Opinions)
    if U is not None:
        plt.plot(T, 1-U.mean(axis=1), '.-', color = 'red', label='mean shadow ban strength')
        plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel("Time [days]")
    plt.ylabel("Opinion")
    
    
def plot_opinion_quantiles(T ,Opinions, q=[50, 25, 75, 5, 95], U = None):
    quantiles = np.percentile(Opinions, q=q, axis=1)

    # Plot quantiles
    if U is not None:
        plt.plot(T, 1-U.mean(axis=1), '.-', color = 'red', label='mean shadow ban strength')
        plt.legend(loc='upper right')
    
    plt.plot(T, quantiles[0], color='black', label='Median Opinion')
    plt.fill_between(T, quantiles[1], quantiles[2], color='blue', alpha=0.5, label='25th-75th Quantiles')
    plt.fill_between(T, quantiles[3], quantiles[4], color='pink', alpha=0.5, label='5th-95th Quantiles')

    plt.xlabel('Time [days]')
    plt.ylabel('Opinion')
    plt.grid()
    
    
def plot_smax_sens(smax_range: np.ndarray, objs_ban: list, means_ban: list):  
    bubble_size = means_ban

    # Normalize bubble size to desired range
    min_size = 10  # Minimum bubble size
    max_size = 100  # Maximum bubble size
    normalized_size = (bubble_size - np.min(bubble_size)) / (np.max(bubble_size) - np.min(bubble_size))
    bubble_size = min_size + normalized_size * (max_size - min_size)
    
    plt.scatter(smax_range, objs_ban, s=bubble_size, c='purple', label='size = mean shadow ban strength')#, alpha=0.5)
    plt.legend(loc='upper right')

    # Connect dots with lines
    plt.plot(smax_range, objs_ban, '-o', color='purple')#, alpha=0.7)
    
    plt.xlim(-0.1, 1.1)
    plt.grid()
    

def plot_density(Opinions_ban):
    # Create a kernel density estimate for each time point
    kde_0 = gaussian_kde(Opinions_ban[0, :])
    kde_Tf = gaussian_kde(Opinions_ban[-1, :])

    # Create an x-axis range for the opinion values
    x = np.linspace(0, 1, 100)

    # Evaluate the density estimates on the x-axis range
    density_0 = kde_0(x)
    density_Tf = kde_Tf(x)

    # Plot the density distributions with filled curves
    plt.plot(x, density_0, label='T$_0$')
    plt.plot(x, density_Tf, label='T$_f$')
    plt.fill_between(x, density_0, alpha=0.3)
    plt.fill_between(x, density_Tf, alpha=0.3)
    plt.legend(loc='upper right')
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
    


