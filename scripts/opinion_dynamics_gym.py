from scipy import integrate
from scipy.integrate import odeint
from scipy.sparse import coo_matrix,diags


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#sign of agent's opinion above or below target
def pull_sign(i, iswitch):
  if i<iswitch:
    psign = -1
  else: 
    psign = 1
  return psign

def clip(x, xmin=0, xmax = 1):
  return min(xmax, max(x,xmin))

 
class Action():
  def __init__(self, opinion, rate=1, targets=None):
    # Initializing the inputted features to the class
    self.opinion = opinion
    self.rate =  rate
    self.targets = targets
  def print(self):
    print(f"Agent opinion = {self.opinion:.3f}\nAgent rate = {self.rate:.3f}")
    print(f"Agent targets = {self.targets}")


class OpinionSimulatorContinuous():
  def __init__(self, rate, shift_function, G, opinions_initial, dt, max_steps = 1000):
    # Initializing the inputted features to the class
    self.A =  nx.adjacency_matrix(G)
    self.A = self.A.tocoo()
    self.num_nodes = self.A.shape[0]
    self.rate = rate
    self.dt = dt
    self.shift = shift_function
    self.max_steps = max_steps
    # Counts the number of steps in a given simulation 
    self.step_counter = 0
    self.opinions_initial = opinions_initial.copy()
    self.opinions = opinions_initial.copy()
    assert len(self.opinions_initial)==self.num_nodes
  
  def slope(self, opinions, action:Action):
    data = self.shift(opinions[self.A.row]- opinions[self.A.col])
    Shift_matrix = coo_matrix((data, (self.A.row, self.A.col)), shape=self.A.shape)
    Rate_matrix = diags(self.rate,0)

    D = Rate_matrix @ Shift_matrix
    Dxdt_no_agent = D.sum(axis = 0).A1  #contribution from following of node
    Dxdt = Dxdt_no_agent  
    
    #contribution from agent
    b = np.zeros(self.num_nodes)
    b[action.targets]= action.rate
    Dxdt_agent = b*self.shift(action.opinion-opinions)  #contribution from agent
    Dxdt += Dxdt_agent
    return Dxdt

  def step(self, action:Action):
    self.step_counter+=1
    mean_old = self.opinions.mean()
    if self.step_counter > self.max_steps:
      done = True
    else: 
      done = False
    #Runge-Kutta step
    state = self.opinions.copy()
    k1 = self.slope(state,action)
    y1 = state + self.dt/2 * k1
    k2 = self.slope(y1,action)
    y2 = state + self.dt/2 * k2
    k3 = self.slope(y2, action)
    y3 = state + self.dt*k3
    k4 = self.slope(y3, action)
    
    #Dxdt = self.slope(action)

    Dxdt_rk = 1/6*(k1 + 2*k2 + 2*k3 + k4)
    self.opinions+= Dxdt_rk*self.dt
    
    #clip opinions at 0 and 1
    self.opinions[self.opinions < 0.0] = 0.0
    self.opinions[self.opinions > 1.0] = 1.0


    reward = state.mean()-mean_old
    return state, done

  def reset(self):
    self.step_counter = 0
    self.opinions = self.opinions_initial.copy()
    state = self.opinions_initial.copy()
    node = None
    done = False
    reward = 0
    return state, done

class AdjointSimulatorContinuous():
  def __init__(self, rate, dshift_function, G, P_final, dt, max_steps = 1000):
    # Initializing the inputted features to the class
    self.A =  nx.adjacency_matrix(G)
    self.A = self.A.tocoo()
    self.num_nodes = self.A.shape[0]
    self.rate = rate
    self.dt = dt
    self.dshift = dshift_function
    self.max_steps = max_steps
    # Counts the number of steps in a given simulation 
    self.step_counter = 0
    self.P_final = P_final.copy()
    self.P = P_final.copy()
    assert len(self.P_final)==self.num_nodes

  def step(self, action:Action, opinions):
    self.step_counter+=1
    if self.step_counter > self.max_steps:
      done = True
    else: 
      done = False

    ddata = self.dshift(opinions[self.A.row]- opinions[self.A.col])
    dShift_matrix = coo_matrix((ddata, (self.A.row, self.A.col)), shape = self.A.shape)
    Rate_matrix = diags(self.rate,0)

    dD = Rate_matrix @ dShift_matrix
    dd = dD.sum(axis=0).A1
    L = self.P*dd #contribution from following of node (its Leaders)
    F = dD @ self.P  #contribution from followers of node (its Followers)
    Dpdt = L-F

    #contribution from agent
    b = np.zeros(self.num_nodes)
    b[action.targets]= action.rate
    LA = self.P*b*self.dshift(action.opinion-opinions) #contribution from agent (its Leader Agent)
    Dpdt += LA
    self.P -= Dpdt*self.dt  #we are going backwards in time
    
    state = self.P.copy()
    return state, done

  def reset(self):
    self.step_counter = 0
    self.P = self.P_final.copy()
    state = self.P_final.copy()
    done = False
    return state, done

class OpinionSimulatorDiscrete():
  def __init__(self, rate, shift_function, G, opinions_initial, max_steps = 1000):
    # Initializing the inputted features to the class
    self.A =  nx.adjacency_matrix(G)
    self.num_nodes = self.A.shape[0]
    self.rate = rate
    self.rate_total = np.sum(rate)
    self.shift = shift_function
    self.max_steps = max_steps
    self.prob = self.rate/self.rate_total
    # Counts the number of steps in a given simulation 
    self.step_counter = 0
    self.opinions_initial = opinions_initial.copy()
    self.opinions = opinions_initial.copy()
    assert len(self.opinions_initial)==self.num_nodes

  def step(self, action:Action):
    self.step_counter+=1
    mean_old = self.opinions.mean()
    if self.step_counter > self.max_steps:
      done = True
    else: 
      done = False
      
    ubot = np.random.uniform(low=0.0, high=1.0)
    pbot = action.rate/(action.rate + self.rate_total) #prob. bot tweets
    if ubot<=pbot:  #bot tweets
      node = 'agent'
      followers = action.targets
      if len(followers)>0:
        diff = action.opinion - self.opinions[followers] 
        #print(f"agent shift = {self.shift(diff)}")
        self.opinions[followers] = self.opinions[followers] + self.shift(diff)
        #clip opinions at 0 and 1
        self.opinions[followers[self.opinions[followers] < 0.0]] = 0.0
        self.opinions[followers[self.opinions[followers] > 1.0]] = 1.0
    else:  #human tweets
      node = np.random.choice(self.num_nodes, p = self.prob)
      followers = self.A[:,node].nonzero()[0]
      if len(followers)>0:
        diff = self.opinions[node] - self.opinions[followers] 
        self.opinions[followers] = self.opinions[followers] + self.shift(diff)
        #clip opinions at 0 and 1
        self.opinions[followers[self.opinions[followers] < 0.0]] = 0.0
        self.opinions[followers[self.opinions[followers] > 1.0]] = 1.0


    state = self.opinions.copy()
    reward = state.mean()-mean_old
    return state, node, reward, done

  def reset(self):
    self.step_counter = 0
    self.opinions = self.opinions_initial.copy()
    state = self.opinions_initial.copy()
    node = None
    done = False
    reward = 0
    return state, node, reward, done