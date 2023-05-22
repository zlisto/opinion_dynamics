import numpy as np
import scipy
import math

from scipy import integrate
from scipy.integrate import odeint
from scipy.sparse import coo_matrix, csr_matrix, diags

import networkx as nx

from typing import List, Set, Dict, Tuple

#######################################################
#shift function f
def shift(x,tau,omega):
    y = omega *x*np.exp(-np.abs(x/tau)**2/2)
    return(y)

#derivative of shift function g
def dshift(x,tau,omega):
    y = omega*(1-np.abs(x/tau)**2)*np.exp(-(x/tau)**2/2)
    return(y)


###############################################################################
def objective_value(OBJECTIVE:str,Opinions:np.ndarray):
    if OBJECTIVE == "MEAN":
        objective = np.mean(Opinions[-1,:])  #maximize mean
    elif OBJECTIVE == "VAR":
        objective = np.var(Opinions[-1,:])  #maximize variance
    elif OBJECTIVE == "VAR_NEG":
        objective = -np.var(Opinions[-1,:])  #maximize variance
    elif OBJECTIVE == "MEAN_TAVG":
        objective = Opinions.mean()  #time avg mean
    elif OBJECTIVE == "VAR_TAVG":
        objective = Opinions.var(axis = 1).mean() #time avg variance
    elif OBJECTIVE == "VAR_TAVG_NEG":
        objective = -Opinions.var(axis = 1).mean() #neg time avg variance
    return objective


#Boundary condition on adjoint and objectives for maximizing 
def boundary_condition_Pf(OBJECTIVE:str, Opinions:np.ndarray):
    n = Opinions.shape[1]  #number of nodes in network (not counting agents)
    if OBJECTIVE == "MEAN":
        Pf = np.ones(n) #final adjoint value for mean objective
    elif OBJECTIVE == "VAR":
        Pf = (Opinions[-1,:] - np.mean(Opinions[-1,:])) #final variance
    elif OBJECTIVE == "VAR_NEG":
        Pf = -(Opinions[-1,:] - np.mean(Opinions[-1,:]))  #final neg variance
    elif OBJECTIVE == "MEAN_TAVG":
        Pf = np.zeros(n)  #time avg mean
    elif OBJECTIVE == "VAR_TAVG":
        Pf = np.zeros(n) #time avg variance
    elif OBJECTIVE == "VAR_TAVG_NEG":
        Pf = np.zeros(n) #neg time avg variance
    else:
        Pf = None
        print(f"Objective {OBJECTIVE} not valid")
    return Pf



#########################################################################################
#calculate the optimal shadow ban matrix Ustar, with elements 0<=ustar<=1
def shadowban_opt(opinions:np.ndarray, ps:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                  tau:float, omega:float, alpha:float, tmax:float):
    assert tmax >0
    ne = A.sum()

    data = shift(opinions[A.row]- opinions[A.col],tau,omega) #shift value
    Shift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape)
    C = Shift_matrix.multiply(np.outer(np.array(rates), ps))
    C = C.multiply(A)
    ne = A.sum()

    nonzero_elements = C.data  # Get the non-zero elements of C
    result = (nonzero_elements >= -alpha/ne/tmax).astype(int) # Perform element-wise comparison

    row_indices, col_indices = C.nonzero()
    Ustar = csr_matrix((result, (row_indices, col_indices)), shape=C.shape)

    return Ustar

############################################################
#Opinion simulator using odeint

#time derivative of opinions
def step_fast_opinion(opinions:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, tau:float, omega:float, 
                      U = None):
    n = len(rates)
    data = shift(opinions[A.row]- opinions[A.col],tau,omega) #shift value
    Shift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) 
    #scale rates on edges by shadow banning policy values
    if U != None:
        Shift_matrix = Shift_matrix.multiply(U)
        
    Rate_matrix = diags(rates,0) #create a diagonal matrix with Rates values

    D = Rate_matrix @ Shift_matrix # matrix multiply
    Dxdt = D.sum(axis = 0).A1 #contribution from following of node

    return Dxdt

#time derivative of opinions for odeint
def dxdt(opinions:np.ndarray, t:float, rates:List, A, tau:float, omega:float):    
    return step_fast_opinion(opinions, rates, A, tau, omega)

#time derivative of opinions with optimal shadowbanning  for odeint (needs adjoints)
def dxdt_opt(opinions:np.ndarray, t:float, rates:List, A, tau:float, omega:float, 
             P:np.ndarray, nsteps:int, tmax:float,  alpha:float):
    T = np.linspace(0, tmax, nsteps)  
    if t<=0:
        tind = 1
    elif t>=tmax:
        tind = nsteps-1
    else:
        tind = np.argmax(T>=t)  #time index in Opinions and agents_opinions of time t
    ps = P[tind,:]
    Ustar = shadowban_opt(opinions, ps, rates, A, tau, omega, alpha, tmax)
    return step_fast_opinion(opinions, rates, A, tau, omega, Ustar)


def simulate_opinion(opinions0:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                         nsteps:int, tmax:float, tau:float, omega:float):
    # Set the initial condition and time points for the integration
    x0 = opinions0
    T = np.linspace(0, tmax, nsteps)  

    # Solve the differential equation using odeint    
    Opinions = odeint( dxdt, x0, T, args=(rates, A, tau, omega))

    return Opinions, T

def simulate_opinion_opt(opinions0:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                         nsteps:int, tmax:float, tau:float, omega:float, 
                          P:np.ndarray, alpha:float):
    # Set the initial condition and time points for the integration
    x0 = opinions0
    T = np.linspace(0, tmax, nsteps)  

    # Solve the differential equation using odeint    
    Opinions = odeint( dxdt_opt, x0, T, args=(rates, A, tau, omega, P, nsteps, tmax, alpha))

                           
    return Opinions, T



#######################################################################################
#Simulate adjoint using numeric integration
#Adjoint time derivative
def step_fast_adjoint(opinions:np.ndarray, ps:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
               tau:float, omega:float, tmax:float, OBJECTIVE:str, U = None):
    n = len(rates)
    ddata = dshift(opinions[A.row]- opinions[A.col],tau,omega) #dshift value
    dShift_matrix = coo_matrix((ddata, (A.row, A.col)), shape=A.shape) #create dshift matrix in coordinate format (row index, col index, value)
    if U != None:
        dShift_matrix = dShift_matrix.multiply(U)
    Rate_matrix = diags(rates,0) #create a diagonal matrix with Rates values

    dD = Rate_matrix @ dShift_matrix
    dd = dD.sum(axis=0).A1
    L = ps*dd #contribution from following of node (its Leaders)
    F = dD @ ps  #contribution from followers of node (its Followers)
   
    Dpdt = L-F
    
    if OBJECTIVE == "MEAN_TAVG":
        Dpdt+=  -np.ones(n)/tmax/n
    elif OBJECTIVE == "VAR_TAVG":
        Dpdt+=  -2*(1-1/n)*(opinions - opinions.mean())/tmax/n
        
    elif OBJECTIVE == "VAR_TAVG_NEG":
        Dpdt+=  2*(1-1/n)*(opinions - opinions.mean())/tmax/n
    
    return Dpdt

#time reverse derivative for numeric integrator odeint
def dpdt_rev(ps, t, Opinions, rates, A, nsteps, tmax, tau:float, omega:float,  OBJECTIVE:str, U = None):
    #assert t >= 0 and t <= tmax, f"t = {t} is out of bounds."
    T = np.linspace(0, tmax, nsteps)  
    if t<=0:
        tind = 1
    elif t>=tmax:
        tind = nsteps-1
    else:
        tind = np.argmax(T>=t)  #time index in Opinions and agents_opinions of time t
    opinions = Opinions[tind,:]
    return step_fast_adjoint(opinions, ps, rates, A, tau, omega, tmax, OBJECTIVE, U)



 
def simulate_adjoint(pf:np.ndarray, Opinions:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                     nsteps:int, tmax:float, tau:float, omega:float,  OBJECTIVE:str, U = None):
    
    Trev = np.linspace(tmax, 0 ,nsteps)  #reversed time because adjoint is simulated backwards  

    # Solve the differential equation using odeint
    args = (Opinions, rates, A, nsteps, tmax, tau, omega, OBJECTIVE, U)
    Prev = odeint(dpdt_rev, pf, Trev, args=args)
    T = Trev[::-1]
    P = Prev[::-1, :]
    return P, T


















#######################################################################################
#Opinion simulation using stiff numerical integrator
def dfun(t,opinions,args):
    rates, A, tau, omega = args
    return step_fast_opinion(opinions, rates, A, tau, omega)

def jacfun(t, opinions, args):
    rates, A, tau, omega = args
    n = len(opinions)
    data = dshift(opinions[A.row]- opinions[A.col],tau,omega) #shift value
    dShift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
    Rate_matrix = diags(rates,0) #create a diagonal matrix with Rates values

    jac = Rate_matrix @ dShift_matrix # matrix multiply
    Dsum = jac.sum(axis = 0).A1 #contribution from following of node
        
    for i in range(n):
        jac[i,i] =  -Dsum[i]
        
    return jac

def simulate_opinion_stiff(opinions0:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                      nsteps:int, tmax:float, tau:float, omega:float):
    
    initial_conditions = opinions0  # Initial values of y
    T = np.linspace(0,tmax,nsteps)  # Time points for which to obtain the solution
    args = (rates, A, tau, omega)

    solver = ode(dfun, jacfun).set_integrator('vode', method='bdf')
    solver.set_initial_value(initial_conditions, T[0]).set_f_params(args).set_jac_params(args)

    solution = [initial_conditions]  # Store the initial conditions as the first solution point


    for t in T[1:]:
        solver.integrate(t)
        solution.append(solver.y)
    solution = np.array(solution)
    Opinions = solution
    return Opinions, T

##################################################
#Simulate adjoint with stiff integrator
def dfunp(t, ps, args):
    Opinions, rates, A, tau, omega = args
    
    T = np.linspace(0, tmax, nsteps)  
    if t<=0:
        tind = 1
    elif t>=tmax:
        tind = nsteps-1
    else:
        tind = np.argmax(T>=t)  #time index in Opinions and agents_opinions of time t
    
    agents_opinion = agents_opinions[tind,:]
    opinions = Opinions[tind,:]
    return step_fast_adjoint(opinions, ps, rates, A, tau, omega)

def jacfunp(t, ps, args):
    Opinions, rates, A, agents_opinions, agents_rate, agents_targets_indices, tau, omega = args
    
    T = np.linspace(0, tmax, nsteps)  
    if t<=0:
        tind = 1
    elif t>=tmax:
        tind = nsteps-1
    else:
        tind = np.argmax(T>=t)  #time index in Opinions and agents_opinions of time t
    agents_opinion = agents_opinions[tind,:]
    opinions = Opinions[tind,:]
    
    n = len(opinions)
    data = dshift(opinions[A.row]- opinions[A.col],tau,omega) #shift value
    dShift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
    Rate_matrix = diags(rates,0) #create a diagonal matrix with Rates values

    jac = Rate_matrix @ dShift_matrix # matrix multiply
    Dsum = jac.sum(axis = 0).A1 #contribution from following of node
    jac = -jac.T
    for (agent_opinion, agent_rate, agent_targets_indices) in zip(agents_opinion, agents_rate, agents_targets_indices):
        b = np.zeros(n)
        b[list(agent_targets_indices)]= agent_rate
        Dsum_agent = b*dshift(agent_opinion-opinions,tau,omega)  #contribution from agent
        Dsum += Dsum_agent
        
    #for i in range(n):
    #    jac[i,i] =  Dsum[i]
        
    return jac

def simulate_adjoint_stiff(pf:np.ndarray, Opinions:np.ndarray, rates:List, A:scipy.sparse.coo_matrix, 
                           nsteps:int, tmax:float, tau:float, omega:float):
    
    initial_conditions = pf  # Initial values of y
    Trev = np.linspace(tmax,0,nsteps)  # Time points for which to obtain the solution
    args = (Opinions, rates, A, agents_opinions, agents_rate, agents_targets_indices, tau, omega)

    #solver = ode(dfunp, jacfunp).set_integrator('vode', method='bdf')
    #solver.set_initial_value(initial_conditions, Trev[0]).set_f_params(args).set_jac_params(args)

    solver = ode(dfunp).set_integrator('vode', method='bdf')
    solver.set_initial_value(initial_conditions, Trev[0]).set_f_params(args)

    solution = [initial_conditions]  # Store the initial conditions as the first solution point


    for t in Trev[1:]:
        solver.integrate(t)
        solution.append(solver.y)
    solution = np.array(solution)
    Prev = solution
    T = Trev[::-1]
    P = Prev[::-1, :]
    return P, T