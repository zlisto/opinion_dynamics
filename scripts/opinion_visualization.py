
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



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
  plt.title(f"R(T) = {reward:.2f}")
  plt.legend()
 
  return fig