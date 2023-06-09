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
  if i < iswitch:
    psign = -1
  else:
    psign = 1
  return psign

def clip(x, xmin=0, xmax=1):
  return min(xmax, max(x,xmin))

# agent action
class Action():
  def __init__(self, user_id, opinion, rate=1, targets=None):
    # Initializing the inputted features to the class
    self.user_id = user_id
    self.opinion = opinion
    self.rate =  rate
    self.targets = targets
  def print(self):
    print(f"Agent id = {self.user_id}\nAgent opinion = {self.opinion:.3f}\nAgent rate = {self.rate:.3f}")
    print(f"Agent targets = {self.targets}")

class OpinionSimulatorContinuous():
  def __init__(self, rate, shift_function, G, opinions_initial, dt, max_steps,bot,agent):
    # Initializing the inputted features to the class
    self.A =  nx.adjacency_matrix(G)
    self.A = self.A.tocoo()
    self.num_nodes = self.A.shape[0]
    self.rate = rate
    self.dt = dt
    self.shift = shift_function
    self.max_steps = max_steps
    self.bot = bot
    self.agent = agent
    # Counts the number of steps in a given simulation 
    self.step_counter = 0
    self.opinions_initial = opinions_initial.copy()
    self.opinions = opinions_initial.copy()
    assert len(self.opinions_initial)==self.num_nodes
  
  def slope(self, opinions, bots_action:list, agents_action:list):
    data = self.shift(opinions[self.A.row]- opinions[self.A.col])
    Shift_matrix = coo_matrix((data, (self.A.row, self.A.col)), shape=self.A.shape)
    Rate_matrix = diags(self.rate,0)

    #contribution from following of nodes
    D = Rate_matrix @ Shift_matrix
    Dxdt_no_agent = D.sum(axis = 0).A1 #.sum(axis=0) returns the sum of each column of a matrix, while .A1 flattens the resulting matrix into a 1D array
    Dxdt = Dxdt_no_agent
    
    #contribution from bots
    if self.bot == 1:
      for bot_action in bots_action:#iterate each bot
        b = np.zeros(self.num_nodes)
        b[bot_action.targets]= bot_action.rate
        Dxdt_bots = b*self.shift(bot_action.opinion-opinions)#contribution from bots
        Dxdt += Dxdt_bots
        
    #contribution from agent
    if self.agent == 1:
      for agent_action in agents_action:#iterate each agent
        b = np.zeros(self.num_nodes)
        b[agent_action.targets]= agent_action.rate
        Dxdt_agent = b*self.shift(agent_action.opinion-opinions)#contribution from agent
        Dxdt += Dxdt_agent

    #stubborn users
    for stub_index in stub_indices:
      Dxdt[stub_index] = 0

    return Dxdt

  def step(self, bots_action:list, agents_action:list):
    self.step_counter+=1
    if self.step_counter >= self.max_steps - 1:
      done = True
    else: 
      done = False

    #Runge-Kutta step
    state = self.opinions.copy()
    k1 = self.slope(state,bots_action,agents_action)
    y1 = state + self.dt/2*k1
    k2 = self.slope(y1,bots_action,agents_action)
    y2 = state + self.dt/2*k2
    k3 = self.slope(y2,bots_action,agents_action)
    y3 = state + self.dt*k3
    k4 = self.slope(y3,bots_action,agents_action)
    Dxdt_rk = (k1 + 2*k2 + 2*k3 + k4)/6
    self.opinions += Dxdt_rk*self.dt
    #clip opinions at 0 and 1
    self.opinions[self.opinions < 0.0] = 0.0
    self.opinions[self.opinions > 1.0] = 1.0

    state = self.opinions.copy()
    rk = [y1,y2,y3]

    return state, done, rk

  def reset(self):
    self.step_counter = 0
    self.opinions = self.opinions_initial.copy()
    state = self.opinions_initial.copy()
    node = None
    done = False
    reward_incr = 0
    return state, done

class AdjointSimulatorContinuous():
  def __init__(self, rate, dshift_function, G, P_final, dt, max_steps,bot,agent):
    # Initializing the inputted features to the class
    self.A =  nx.adjacency_matrix(G)
    self.A = self.A.tocoo()
    self.num_nodes = self.A.shape[0]
    self.rate = rate
    self.dt = dt
    self.dshift = dshift_function
    self.max_steps = max_steps
    self.bot = bot
    self.agent = agent
    # Counts the number of steps in a given simulation 
    self.step_counter = 0
    self.P_final = P_final.copy()
    self.P = P_final.copy()
    assert len(self.P_final)==self.num_nodes

  def slope(self, opinions, P, bots_action:list, agents_action:list):
    ddata = self.dshift(opinions[self.A.row]- opinions[self.A.col])
    dShift_matrix = coo_matrix((ddata, (self.A.row, self.A.col)), shape = self.A.shape)
    Rate_matrix = diags(self.rate,0)

    dD = Rate_matrix @ dShift_matrix
    dd = dD.sum(axis=0).A1 #.sum(axis=0) returns the sum of each column of a matrix, while .A1 flattens the resulting matrix into a 1D array
    L = self.P*dd #contribution from following of node (its Leaders)
    F = dD @ self.P  #contribution from followers of node (its Followers)
    Dpdt = L-F

    #contribution from bots
    if self.bot == 1:
      for bot_action in bots_action:#iterate each bot
        b = np.zeros(self.num_nodes)
        b[bot_action.targets]= bot_action.rate
        LA = self.P*b*self.dshift(bot_action.opinion-opinions) #contribution from bots
        Dpdt += LA

    #contribution from agent
    if self.agent == 1:
      for agent_action in agents_action:#iterate each agent
        b = np.zeros(self.num_nodes)
        b[agent_action.targets]= agent_action.rate
        LA = self.P*b*self.dshift(agent_action.opinion-opinions) #contribution from agent (its Leader Agent)
        Dpdt += LA

    #stubborn users
    for stub_index in stub_indices:
      Dpdt[stub_index] = 0    

    return Dpdt

  def step(self, agents_action:list, opinion, opinion_RK):
    self.step_counter+=1
    if self.step_counter >= self.max_steps - 1:
      done = True
    else: 
      done = False

    #Runge-Kutta step
    state = self.P.copy()
    k1 = self.slope(opinion,state,bots_action,agents_action)
    y1 = state + self.dt/2*k1
    k2 = self.slope(opinion_RK[0],y1,bots_action,agents_action)
    y2 = state + self.dt/2*k2
    k3 = self.slope(opinion_RK[1],y2,bots_action,agents_action)
    y3 = state + self.dt*k3
    k4 = self.slope(opinion_RK[2],y3,bots_action,agents_action)
    Dpdt_rk = (k1 + 2*k2 + 2*k3 + k4)/6

    self.P -= Dpdt_rk*self.dt #we are going backwards in time
    state = self.P.copy()

    return state, done

  def reset(self):
    self.step_counter = 0
    self.P = self.P_final.copy()
    state = self.P_final.copy()
    done = False
    return state, done
