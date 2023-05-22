
import os
import numpy as np
from scipy import integrate
from scipy.sparse import coo_matrix,diags
import scipy
import math
import matplotlib.pyplot as plt
import networkx as nx

import json

import matplotlib.cm as cm
import pandas as pd

from typing import List, Set, Dict, Tuple
from scipy.integrate import odeint

#######################################################
#shift function f
def shift(x,tau,omega):
    y = omega *x*np.exp(-np.abs(x/tau)**2/2)
    #y = omega*x*(np.heaviside(x+tau,1)-np.heaviside(x-tau,0))
    return(y)

#derivative of shift function g
def dshift(x,tau,omega):
    y = omega*(1-np.abs(x/tau)**2)*np.exp(-(x/tau)**2/2)
    return(y)


###############################################################################
#Boundary condition on adjoint and objectives
def boundary_condition_Pf(OBJECTIVE:str, Opinions:np.ndarray, opinion_target = 0.5):
    n = Opinions.shape[1]  #number of nodes in network (not counting agents)
    if OBJECTIVE == "MEAN":
        Pf = -np.ones(n) #final adjoint value for mean objective
    elif OBJECTIVE == "VARIANCE":
        Pf = -(Opinions[-1,:] - np.mean(Opinions[-1,:])) #final adjoint value for variance objective
    elif OBJECTIVE == "TARGET":
        Pf = (Opinions[-1,:] - opinion_target) #final adjoint value for target objective
    return Pf

def objective_value(OBJECTIVE:str,Opinions:np.ndarray,opinion_target=0.5):
    if OBJECTIVE == "MEAN":
        objective = -np.mean(Opinions[-1,:])  #maximize mean
    elif OBJECTIVE == "VARIANCE":
        objective = -np.var(Opinions[-1,:])  #maximize variance
    elif OBJECTIVE == "TARGET":
        objective = np.sqrt(np.mean((Opinions[-1,:]-opinion_target)**2))  #hit a target opinion
    return objective



#############################################################################
#Functions to calculate optimal agent policy for shooting (minimize Hamiltonian)
def hamiltonian_agent(u,opinions,ps,tau,omega):
    q = shift(u-opinions,tau,omega)
    q = q.T
    return np.dot(ps,q)

def agent_opt(opinions,ps,tau,omega):
    # Define the minimum and maximum values of u
    umin = np.min(opinions)-2*tau
    umax = np.max(opinions)+2*tau
    urange = np.linspace(umin, umax)
    # Define the bounds for u
    fstar = float('inf')
    ustar = umin
    ps_norm = ps / np.linalg.norm(ps)  #normalize p array so it has unit norm (in case p is too big)
    for u in urange:
        f = hamiltonian_agent(u,opinions,ps_norm,tau,omega)
        if f<fstar:
            fstar = f
            ustar = u
    return ustar

def agent_opt_min(opinions,ps,tau,omega):
    ustar = opinions[np.argmin(ps)]+tau
    return ustar

#######################################################################################
#Derivatives for opinions 
def step_fast_opinion(opinions:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
              agents_opinion:List, agents_rate:List, agents_targets_indices:List, tau:float, omega:float):
    n = len(rates)
    data = shift(opinions[A.row]- opinions[A.col],tau,omega) #shift value
    Shift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
    Rate_matrix = diags(rates,0) #create a diagonal matrix with Rates values

    D = Rate_matrix @ Shift_matrix # matrix multiply
    Dxdt_no_agent = D.sum(axis = 0).A1 #contribution from following of node
    Dxdt = Dxdt_no_agent

    for (agent_opinion, agent_rate, agent_targets_indices) in zip(agents_opinion, agents_rate, agents_targets_indices):
        b = np.zeros(n)
        b[list(agent_targets_indices)]= agent_rate
        Dxdt_agent = b*shift(agent_opinion-opinions,tau,omega)  #contribution from agent
        Dxdt += Dxdt_agent
    return Dxdt

#Derivatives for opinions 
def step_fast_opinion_no_agent(opinions:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, tau:float, omega:float):
    n = len(rates)
    data = shift(opinions[A.row]- opinions[A.col],tau,omega) #shift value
    Shift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
    Rate_matrix = diags(rates,0) #create a diagonal matrix with Rates values

    D = Rate_matrix @ Shift_matrix # matrix multiply
    Dxdt_no_agent = D.sum(axis = 0).A1 #contribution from following of node
    Dxdt = Dxdt_no_agent

    return Dxdt

def dxdt(opinions, t, rates, A, agents_opinion, agents_rate, agents_targets_indices, tau:float, omega:float):
    return step_fast_opinion(opinions, rates, A, agents_opinion, agents_rate, agents_targets_indices, tau, omega)


def dxdt_pace_mean(opinions, t,rates, A, agents_rate, agents_targets_indices, tau:float, omega:float):
    agents_opinion=[]
    for targets_indices in agents_targets_indices:
        agent_opinion = np.mean(opinions[targets_indices]) + tau  #agent is conf. bound away from follower
        agent_opinion =min(max(agent_opinion,0),1)
        agents_opinion.append(agent_opinion)
    return step_fast_opinion(opinions, rates, A, agents_opinion, agents_rate, agents_targets_indices, tau, omega)

def dxdt_opt(opinions, t, P, rates, A, agents_rate, agents_targets_indices, nsteps, tmax, tau:float, omega:float):
    T = np.linspace(0, tmax, nsteps)  
    if t<=0:
        tind = 1
    elif t>=tmax:
        tind = nsteps-1
    else:
        tind = np.argmax(T>=t)  #time index in Opinions and agents_opinions of time t
    ps = P[tind,:]
    agents_opinion=[]
    for agent_targets_indices in agents_targets_indices:
        agent_opinion = agent_opt_min(opinions[agent_targets_indices],ps[agent_targets_indices],tau,omega)  #agent minimizes Hamiltonian (if goal is to min. objective)
        agent_opinion =min(max(agent_opinion,0),1)
        agents_opinion.append(agent_opinion)
    return step_fast_opinion(opinions, rates, A, agents_opinion, agents_rate, agents_targets_indices, tau, omega)



#######################################################################################
#Opinion simulation using numerical integrator

def simulate_opinion(opinions0:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                     agents_opinions:np.ndarray, agents_rates:List, 
                     agents_targets_indices:List[List], nsteps:int, tmax:float, tau:float, omega:float):
    # Set the initial condition and time points for the integration
    x0 = opinions0
    T = np.linspace(0, tmax, nsteps)  

    # Solve the differential equation using odeint
    agents_opinions = []
    Opinions = odeint( dxdt, x0, T, args=(rates, A, agents_opinions ,agents_rates, agents_targets_indices, tau, omega))
    return Opinions, T

def simulate_opinion_pace_mean(opinions0:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                     agents_rate:List, agents_targets_indices:List[List], 
                               nsteps:int, tmax:float, tau:float, omega:float):
    # Set the initial condition and time points for the integration
    x0 = opinions0
    T = np.linspace(0, tmax, nsteps)  

    # Solve the differential equation using odeint    
    Opinions = odeint( dxdt_pace_mean, x0, T, args=(rates, A, agents_rate, agents_targets_indices, tau, omega))
    #calculate agent opinion using human opinions and policy rule
    agents_opinions=[]
    for targets_indices in agents_targets_indices:
        agent_opinion = np.mean(Opinions[:,targets_indices], axis = 1)+ tau  #policy rule: agent is conf. bound away from follower
        agent_opinion =np.minimum(np.maximum(agent_opinion,0),1)
        agents_opinions.append(agent_opinion)
        agents_opinions = np.array( agents_opinions)
    return Opinions, T, agents_opinions.T



def simulate_opinion_opt(opinions0:np.ndarray, P:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                     agents_rate:List, agents_targets_indices:List[List], 
                         nsteps:int, tmax:float, tau:float, omega:float):
    # Set the initial condition and time points for the integration
    x0 = opinions0
    T = np.linspace(0, tmax, nsteps)  

    # Solve the differential equation using odeint    
    Opinions = odeint( dxdt_opt, x0, T, args=(P,rates, A, agents_rate, agents_targets_indices, nsteps, tmax, tau, omega))
    #calculate agent opinion using human opinions and policy rule
    nagents = len(agents_rate)
    agents_opinions = np.zeros((nsteps, nagents))
    c=0
    for agent_targets_indices in agents_targets_indices:
        agent_opinions = np.zeros((nsteps,1))
        for tind in range(nsteps):
            agent_opinion = agent_opt_min(Opinions[tind,agent_targets_indices],P[tind,agent_targets_indices],tau,omega)  
            agent_opinion =np.minimum(np.maximum(agent_opinion,0),1)
            agent_opinions[tind,:]  = agent_opinion
        agents_opinions[:,c] = agent_opinions[:,0]
        c+=1
    return Opinions, T, agents_opinions
###############################################################################3
#Derivative of Adjoint
def step_fast_adjoint(opinions:np.ndarray, ps:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
              agents_opinion:List, agents_rate:List, agents_targets_indices:List, tau:float, omega:float):
    n = len(rates)
    ddata = dshift(opinions[A.row]- opinions[A.col],tau,omega) #dshift value
    dShift_matrix = coo_matrix((ddata, (A.row, A.col)), shape=A.shape) #create dshift matrix in coordinate format (row index, col index, value)
    Rate_matrix = diags(rates,0) #create a diagonal matrix with Rates values

    dD = Rate_matrix @ dShift_matrix
    dd = dD.sum(axis=0).A1
    L = ps*dd #contribution from following of node (its Leaders)
    F = dD @ ps  #contribution from followers of node (its Followers)
   
    Dpdt = L-F

    for (agent_opinion, agent_rate, agent_targets_indices) in zip(agents_opinion, agents_rate, agents_targets_indices):
        b = np.zeros(n)
        b[list(agent_targets_indices)]= agent_rate
        Dpdt_agent = ps*b*dshift(agent_opinion-opinions,tau,omega) #contribution from agent (its Leader Agent)
        Dpdt += Dpdt_agent
    return Dpdt


def dpdt_rev(ps, t, Opinions, rates, A, agents_opinions, agents_rate, agents_targets_indices, 
             nsteps, tmax, tau:float, omega:float):
    #assert t >= 0 and t <= tmax, f"t = {t} is out of bounds."
    T = np.linspace(0, tmax, nsteps)  
    if t<=0:
        tind = 1
    elif t>=tmax:
        tind = nsteps-1
    else:
        tind = np.argmax(T>=t)  #time index in Opinions and agents_opinions of time t
    agents_opinion = agents_opinions[tind,:]
    opinions = Opinions[tind,:]
    return step_fast_adjoint(opinions, ps, rates, A, agents_opinion, agents_rate, agents_targets_indices, tau, omega)

#######################################################################################
#Simulate adjoint using numeric integration
 
def simulate_adjoint(pf:np.ndarray, Opinions:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                     agents_opinions:np.ndarray, agents_rate:List, agents_targets_indices:List[List], 
                     nsteps:int, tmax:float, tau:float, omega:float):
    
    Trev = np.linspace(tmax, 0 ,nsteps)  #reversed time because adjoint is simulated backwards  

    # Solve the differential equation using odeint
    Prev = odeint(dpdt_rev, pf, Trev, args=(Opinions, rates, A, agents_opinions , agents_rate, agents_targets_indices,
                                       nsteps, tmax, tau, omega))
    T = Trev[::-1]
    P = Prev[::-1, :]
    return P, T