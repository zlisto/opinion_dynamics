import numpy as np
from typing import List, Set, Dict, Tuple
import networkx as nx
import scipy
import math
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix,diags

omega = 1 # strength of persuasion
tau = 0.1 # confidence interval/easiness to pursuade
confidence_bound_decay = 1
# shift function: f
def shift(x,tau=0.1,omega=1):
    y = omega*x*np.exp(-x**2/tau**2/2)
    return(y)

# derivative of shift function: g
def dshift(x,tau=0.1,omega=1):
    y = omega*(1-x**2/tau**2)*np.exp(-x**2/tau**2/2)
    return(y)

def step_fast_opinions(Opinions:np.ndarray, Rates:List, A:scipy.sparse.coo.coo_matrix, 
              agent_opinions:List, agent_rates:List, targets_indices:List[List]):
    # Impacts from HUMAN nodes
    Rate_matrix = diags(Rates,0) #create a diagonal matrix with Rates values
    data = shift(Opinions[A.row]- Opinions[A.col]) #shift value
    Shift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
    D = Rate_matrix @ Shift_matrix # matrix multiply
    Dxdt_human = D.sum(axis = 0).A1 # impacts from following HUMAN nodes
    Dxdt = Dxdt_human

    # Impacts from agent nodes
    n = len(Rates)
    # Loop for each agent
    for (agent_opinion, agent_rate, targets_index) in zip(agent_opinions, agent_rates, targets_indices):
        rx_rate = np.zeros(n) # receiving rate
        rx_rate[list(targets_index)]= agent_rate
        Dxdt_agent = rx_rate*shift(agent_opinion - Opinions,tau,omega) # impacts from following an agent
        Dxdt += Dxdt_agent

    return Dxdt

# ts = t_star = switching time
def simulate_opinion_agent_one_follower(Opinions0:np.ndarray, Rates:List, A:scipy.sparse.coo.coo_matrix, 
                     agent_rates:List, targets_indices:List[List], nsteps:int, tstep:float, ts:float, z:int):
    # Declare local variables
    n =len(Opinions0) # number of Humans and agents nodes
    nagents = len(targets_indices) # number of image agents
    Opinions = np.zeros((nsteps,n)) # opinions of HUMAN and agent nodes
    image_agents_opinions = np.zeros((nsteps,nagents)) # opinions of image agents
    # zigzag assumptions
    Tp = 8
    Tps = 1/4

    # Iterate for nsteps
    for i in range(nsteps):
        t = tstep*i
        if i==0:
            Opinions[i,:] = Opinions0 # initial opinions of all nodes
        else:
            opinions = Opinions[i-1,:]
            agents_opinion=[]
            na = 0 # index number of image agent
            # Update image agents' opinions
            for targets_index in targets_indices:
                agent_opinion = opinions[targets_index[0]] + tau*np.sign(t-ts)  #agent is conf. bound away from follower
                agent_opinion = min(max(agent_opinion,0),1)
                agents_opinion.append(agent_opinion)
                image_agents_opinions[i-1,na] = agent_opinion
                # zigzag modification: lower highs and higher lows
                if z==1:
                  modulo = (i % Tp)/Tp
                  if i==1:
                    1==1
                  elif t < ts:
                    if modulo <= Tps:
                      agent_opinion = image_agents_opinions[i-2,na] + tau*tstep
                      agent_opinion = min(max(agent_opinion,0),1)
                      agents_opinion[-1] = agent_opinion
                      image_agents_opinions[i-1,na] = agent_opinion
                    elif modulo != 0:
                      agent_opinion = image_agents_opinions[i-2,na] - tau*tstep
                      agent_opinion = min(max(agent_opinion,0),1)
                      agents_opinion[-1] = agent_opinion
                      image_agents_opinions[i-1,na] = agent_opinion
                  else: # t >= ts
                    if modulo <= Tps:
                      agent_opinion = image_agents_opinions[i-2,na] - tau*tstep
                      agent_opinion = min(max(agent_opinion,0),1)
                      agents_opinion[-1] = agent_opinion
                      image_agents_opinions[i-1,na] = agent_opinion
                    elif modulo != 0:
                      agent_opinion = image_agents_opinions[i-2,na] + tau*tstep
                      agent_opinion = min(max(agent_opinion,0),1)
                      agents_opinion[-1] = agent_opinion
                      image_agents_opinions[i-1,na] = agent_opinion
                na+=1
            # Update humans' and agents' opinions
            Dxdt = step_fast_opinions(opinions, Rates, A, agents_opinion, agent_rates, targets_indices)
            opinions_new = opinions+Dxdt*tstep
            opinions_new = np.maximum(np.minimum(opinions_new,np.ones(n)),np.zeros(n))
            Opinions[i,:] = opinions_new   

    return (Opinions, image_agents_opinions)

def step_fast(Opinions:np.ndarray, P:np.ndarray, Rates:List, A:scipy.sparse.coo.coo_matrix, 
              agent_opinions:List, agent_rates:List, targets_indices:List[List]):
    # Impacts from HUMAN nodes
    Rate_matrix = diags(Rates,0) #create a diagonal matrix with Rates values
    data = shift(Opinions[A.row]- Opinions[A.col],tau,omega) #shift value
    Shift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
    D = Rate_matrix @ Shift_matrix # matrix multiply
    Dxdt_human = D.sum(axis = 0).A1 # impacts from following HUMAN nodes
    Dxdt = Dxdt_human

    ddata = dshift(Opinions[A.row]- Opinions[A.col],tau,omega) #dshift value
    dShift_matrix = coo_matrix((ddata, (A.row, A.col)), shape=A.shape) #create dshift matrix in coordinate format (row index, col index, value)
    dD = Rate_matrix @ dShift_matrix
    dd = dD.sum(axis=0).A1
    L = P*dd #contribution from following of node (its Leaders)
    F = dD @ P  #contribution from followers of node (its Followers)
    Dpdt = L-F

    # Impacts from agent nodes
    n = len(Rates)
    # Loop for each agent
    for (agent_opinion,agent_rate,targets_index) in zip(agent_opinions, agent_rates, targets_indices):
        rx_rate = np.zeros(n) # receiving rate
        rx_rate[list(targets_index)]= agent_rate
        Dxdt_agent = rx_rate*shift(agent_opinion-Opinions,tau,omega)  #contribution from agent
        Dxdt += Dxdt_agent

        LA = P*rx_rate*dshift(agent_opinion-Opinions,tau,omega) #contribution from agent (its Leader Agent)
        Dpdt += LA
    
    return (Dxdt,Dpdt)

def simulate_opinion(Opinions0:np.ndarray, Rates:List, A:scipy.sparse.coo.coo_matrix, 
                     Agent_opinions:np.ndarray, agent_rates:List, 
                     targets_indices:List[List], nsteps:int, tstep:float):
    n =len(Opinions0)
    Opinions = np.zeros((nsteps,n))
    DxDt = np.zeros((nsteps,n))
    # Bug: this P should initilize based on boundary condition
    # P = np.zeros(n)
    for i in range(nsteps):    
        if i==0:
            Opinions[i,:] = Opinions0
        else:
            agent_opinions = Agent_opinions[i-1,:]
            opinions = Opinions[i-1,:]
            Dxdt = step_fast_opinions(opinions, Rates, A, agent_opinions, agent_rates, targets_indices)
            # (Dxdt,Dpdt) = step_fast(opinions, P, Rates, A, agent_opinions, agent_rates, targets_indices)
            opinions_new = opinions + Dxdt*tstep
            #opinions_new = np.maximum(np.minimum(opinions_new,np.ones(n)),np.zeros(n))
            Opinions[i,:] = opinions_new
            DxDt[i,:] = Dxdt
    
    return (Opinions,DxDt)

def boundary_condition_Pf(OBJECTIVE:str, Opinions:np.ndarray, opinion_target = 0.5):
    n = Opinions.shape[1]  #number of nodes in network (not counting agents)
    if OBJECTIVE == "MEAN":
        Pf = -np.ones(n) #final adjoint value for mean objective
    elif OBJECTIVE == "TIME_AVG_MEAN":
        Pf = -np.ones(n) #final adjoint value for time average mean objective
        objective = -np.mean(Opinions[-1,:]) - G/nsteps*Opinions.sum()  #maximize mean plus time-average mean
    elif OBJECTIVE == "VARIANCE":
        Pf = -(Opinions[-1,:] - np.mean(Opinions[-1,:])) #final adjoint value for variance objective
    elif OBJECTIVE == "TARGET":
        Pf = (Opinions[-1,:] - opinion_target) #final adjoint value for target objective
    return Pf

def simulate_adjoint_reverse(Pf:np.ndarray, Opinions:np.ndarray, Rates:List, A:scipy.sparse.coo.coo_matrix, 
                             Agent_opinions:np.ndarray, agent_rates:List, Targets_indices:List, 
                             nsteps:int, tstep:float):
    assert Agent_opinions.shape[0]==Opinions.shape[0]
    n =len(Pf)
    P = np.zeros((nsteps,n))

    for i in range(nsteps):  
        if i==0:
            P[nsteps-1-i,:] = Pf
        else:
            agent_opinions = Agent_opinions[nsteps-i,:]
            (Dxdt,Dpdt) = step_fast(Opinions[nsteps-i,:], P[nsteps-i,:], Rates, A, agent_opinions, agent_rates, Targets_indices)
            P[nsteps-1-i,:] = P[nsteps-i,:] - Dpdt*tstep
    return (P)


def objective_value(OBJECTIVE:str,Opinions:np.ndarray,opinion_target=0.5):
    if OBJECTIVE == "MEAN":
        objective = -np.mean(Opinions[-1,:])  #maximize mean
    elif OBJECTIVE == "TIME_AVG_MEAN":
        objective = -np.mean(Opinions[-1,:]) - G/len(Opinions[0])/nsteps*Opinions.sum()  #maximize mean plus time-average mean
    elif OBJECTIVE == "VARIANCE":
        objective = -np.var(Opinions[-1,:])  #maximize variance
    elif OBJECTIVE == "TARGET":
        objective = np.sqrt(np.mean((Opinions[-1,:]-opinion_target)**2))  #hit a target opinion
    return objective

def plot_opinion_P(T,Opinions,P,Agent_opinions):
  # to change to all show mean
  print(f"objective = {objective_value(OBJECTIVE,Opinions):.4f}")

  plt.figure(figsize =(16,4))
  plt.subplot(1,2,1)
  for i in range(len(Opinions[0])):
    plt.plot(T,Opinions[:,i],label=inv_node_index[i])
  plt.plot(T,Agent_opinions,label='agent',color ='red')
  plt.ylabel('Opinion',fontsize = 18)
  plt.xlabel('Time',fontsize = 18)
  plt.ylim([0,1])
  plt.legend(loc="lower right")

  plt.subplot(1,2,2)
  for i in range(len(P[0])):
    plt.plot(T,P[:,i],label=inv_node_index[i])
  plt.ylabel('Adjoint variable',fontsize = 18)
  plt.xlabel('Time',fontsize = 18)
  plt.legend(loc="lower right")
  plt.show()

def plot_opinion_P_quantiles(T,Opinions,Agent_opinion,P):
  Q05 = np.quantile(Opinions,0.05,axis = 1)
  Q95 = np.quantile(Opinions,0.95,axis = 1)
  Q50 = np.quantile(Opinions,0.50,axis = 1)
  Q25 = np.quantile(Opinions,0.25,axis = 1)
  Q75 = np.quantile(Opinions,0.75,axis = 1)

  plt.figure(figsize =(12,6))
  plt.subplot(1,2,1)
  plt.plot(T,Q05,color = 'black')
  plt.plot(T,Q95,color = 'black')
  plt.fill_between(T, Q05, Q95,alpha=0.2)
  plt.plot(T,Q25,color = 'black')
  plt.plot(T,Q75,color = 'black')
  plt.fill_between(T, Q25, Q75,alpha=0.2)
  plt.plot(T,Q50,color = 'blue')

  plt.plot(T,Agent_opinion,marker='.',label='agent',color ='red')
  plt.ylabel('Opinion',fontsize = 18)
  plt.xlabel('Time',fontsize = 18)
  plt.ylim([0,1])
  plt.legend()

  P05 = np.quantile(P,0.05,axis = 1)
  P95 = np.quantile(P,0.95,axis = 1)
  P50 = np.quantile(P,0.50,axis = 1)
  P25 = np.quantile(P,0.25,axis = 1)
  P75 = np.quantile(P,0.75,axis = 1)

  plt.subplot(1,2,2)
  plt.plot(T,P05,color = 'black')
  plt.plot(T,P95,color = 'black')
  plt.fill_between(T, P05, P95,alpha=0.2)
  plt.plot(T,P25,color = 'black')
  plt.plot(T,P75,color = 'black')
  plt.fill_between(T, P25, P75,alpha=0.2)
  plt.plot(T,P50,color = 'blue')

  plt.ylabel('P',fontsize = 18)
  plt.xlabel('Time',fontsize = 18)
  plt.show()

def plot_opinion_quantiles(T:np.ndarray,Opinions:np.ndarray):
    Q05 = np.quantile(Opinions,0.05,axis = 1)
    Q95 = np.quantile(Opinions,0.95,axis = 1)
    Q50 = np.quantile(Opinions,0.50,axis = 1)
    Q25 = np.quantile(Opinions,0.25,axis = 1)
    Q75 = np.quantile(Opinions,0.75,axis = 1)

    plt.plot(T,Q05,color = 'black')
    plt.plot(T,Q95,color = 'black')
    plt.fill_between(T, Q05, Q95,alpha=0.2)
    plt.plot(T,Q25,color = 'black')
    plt.plot(T,Q75,color = 'black')
    plt.fill_between(T, Q25, Q75,alpha=0.2)
    plt.plot(T,Q50,color = 'blue')
    plt.ylabel('Opinion',fontsize = 18)
    plt.xlabel('Time',fontsize = 18)
    plt.ylim([0,1])