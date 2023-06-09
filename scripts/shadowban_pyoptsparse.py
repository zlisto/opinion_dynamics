import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.integrate import odeint
from scipy.sparse import coo_matrix, csr_matrix, diags

import control.optimal as obc
import control as ct
from pyoptsparse import Optimization, OPT
import pyoptsparse as pos
import scipy.sparse as sparse

import networkx as nx
import winsound
import time

from typing import List, Set, Dict, Tuple

#######################################################
#shift function f
def shift(x,tau,omega):
    x = np.clip(x, -1, 1)
    y = omega *x*np.exp(-np.abs(x/tau)**2/2)
    return(y)

#derivative of shift function g
def dshift(x,tau,omega):
    x = np.clip(x, -1, 1)
    y = omega*(1-np.abs(x/tau)**2)*np.exp(-(x/tau)**2/2)
    return(y)




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
def dxdt(opinions:np.ndarray, t:float, rates:List, A, tau:float, omega:float, U = None):    
    return step_fast_opinion(opinions, rates, A, tau, omega, U)


#######################################################################################
##################################################
###################################################################################################################
###################################################################################################################
#control functions
def sys_update(t, x, u, params):
    # Get the parameters for the model
    A = params.get('A')         # vehicle wheelbase
    rates = params.get('rates')
    tau = params.get('tau')
    omega = params.get('omega')
    
    
    # Create COO matrix U with data from u at A's locations
    
    U_row = A.row
    U_col = A.col
    U = coo_matrix((u, (U_row, U_col)), shape=(A.shape[0], A.shape[1]))

    # Return the derivative of the state
    return dxdt(x, t, rates, A, tau, omega, U)

#########################################################
#optimization functions
def extract_input_state(coeffs, nv, ne, npts):
    x = coeffs[-nv * npts:].reshape(nv, -1)
    coeffs = coeffs[:-nv * npts]
    u = coeffs.reshape((ne, -1))
    return u, x

def initial_control(params,OBJECTIVE):
    '''Calculate the initial guess for the control.  Returns an array U0 with size (ne, npts)'''
    A, E, rates, tau, omega, x0 = params['A'], params['E'], params['rates'], params['tau'], params['omega'],params['opinions0']
    npts =  params['npts']
    nv, ne = A.shape[0], E.shape[1]
    #smart initial condition (for maximizing mean)
    grid = np.linspace(0, 1, npts)
    data = shift(x0[A.row]- x0[A.col],tau,omega)
    
    if OBJECTIVE == 'NONE':
        #dumb initial condition
        u0 = np.ones(ne)
        U0 =  np.vstack([u0] * npts).T
    elif  OBJECTIVE == 'MEAN':
        u0_smart = (data>0).astype(int)
        U0= np.outer(u0_smart, 1 - grid) + np.outer(np.ones_like(u0_smart), grid)
    elif  OBJECTIVE == 'VARMAX':
        #smart initial condition (for maximizing var)
        u0_smart = (np.sign(x0[A.col]-x0.mean())*np.sign(data)>0).astype(int)
        U0 = np.outer(u0_smart, 1 - grid) + np.outer(np.ones_like(u0_smart), grid)
    elif  OBJECTIVE == 'VARMIN':
        #smart initial condition (for minimizing var)
        u0_smart = (np.sign(x0[A.col]-x0.mean())*np.sign(data)<=0).astype(int)
        U0 = np.outer(u0_smart, 1 - grid) + np.outer(np.ones_like(u0_smart), grid)
    else:
        print("Error: Wrong Objective.  Choose from NONE, MEAN, VARMAX, VARMIN")
        U0 = None
    return U0

def cost_sim(OBJECTIVE, Opinions, U = None, alpha = 0):
    '''Cost of opinions and control in simulator.'''
    if U is None:
        U = np.ones(Opinions.shape)
    cu = -alpha*U.mean(axis=1)         

    if OBJECTIVE == 'MEAN':
        cx = -Opinions.mean(axis = 1)
    elif  OBJECTIVE == 'VARMAX':
        cx = -Opinions.var(axis = 1)
    elif  OBJECTIVE == 'VARMIN':
        cx = Opinions.var(axis = 1)
    else:
        print("Error: Wrong Objective.  Choose from NONE, MEAN, VARMAX, VARMIN")
        cx = None
    
    return np.mean(cu + cx)
    
def simulate_opinions(params, sys, Ucollocation = None):
    '''Simulate the opinions in sys with parameters in params dictionary with control Ucollocation which has dim ne x npts'''
    A, E, rates, tau, omega, x0 = params['A'], params['E'], params['rates'], params['tau'], params['omega'],params['opinions0']
    npts, npts_eval, Tf, alpha, OBJECTIVE = params['npts'], params['npts_eval'], params['Tf'], params['alpha'], params['OBJECTIVE']
    nv, ne = A.shape[0], E.shape[1]
    
    timepts = np.linspace(0, Tf, npts, endpoint=True)
    t_eval = np.linspace(0, Tf, npts_eval)
    if Ucollocation is None:
        Ucollocation = np.ones((ne, npts)) 
        
    resp = ct.input_output_response(
                            sys, timepts, Ucollocation, x0,
                            t_eval= t_eval, params = params)
    T, Opinions, U = resp.time, resp.outputs, resp.inputs
     
    return T, Opinions.T, U.T


def solve_shadowban(params, optimizer):
    '''Solve shadowban control problem using collocation and pyoptsparse'''
    A, E, rates, tau, omega, x0 = params['A'], params['E'], params['rates'], params['tau'], params['omega'],params['opinions0']
    npts, npts_eval, Tf, alpha, OBJECTIVE = params['npts'], params['npts_eval'], params['Tf'], params['alpha'], params['OBJECTIVE']
    nv, ne = A.shape[0], E.shape[1]
    
    timepts = np.linspace(0, Tf, npts, endpoint=True)
    t_eval = np.linspace(0, Tf, npts_eval)
    dt = timepts[1]-timepts[0]

    sys = ct.NonlinearIOSystem(
        updfcn =sys_update, outfcn= None, states=nv,
        inputs=ne, outputs = nv,
        name='shadowban network', params=params)


    #bounds on control strength
    constraints = [ obc.input_range_constraint(sys, np.zeros(ne), np.ones(ne)), 
                   obc.state_range_constraint(sys, np.zeros(nv), np.ones(nv)) ]   
    

    ###################################################
    #Objective Function
    def cost_mean(x, u):
        return -x.mean()/Tf - alpha*u.mean()/Tf

    #Objective Function
    def cost_varmax(x, u):
        return -x.var()/Tf - alpha*u.mean()/Tf

    #Objective Function
    def cost_varmin(x, u):
        return x.var()/Tf - alpha*u.mean()/Tf
    
    if OBJECTIVE == 'MEAN':
        integral_cost = cost_mean
    elif  OBJECTIVE == 'VARMAX':
        integral_cost = cost_varmax
    elif  OBJECTIVE == 'VARMIN':
        integral_cost = cost_varmin
    else:
        print("Error: Wrong Objective.  Choose from NONE, MEAN, VARMAX, VARMIN")
        integral_cost=-1

    U0_initial_guess = initial_control(params,OBJECTIVE)
    U0_no_agent = initial_control(params,"NONE")

    ocp = obc.OptimalControlProblem(
            sys, timepts, integral_cost, trajectory_constraints=constraints,
            terminal_cost=None, terminal_constraints=[],
            initial_guess=U0_initial_guess)
    ocp.x = x0
    
    resp = ct.input_output_response(
                        sys, timepts, U0_no_agent, x0,
                        t_eval= t_eval, params = params)

    T, Opinions, U = resp.time, resp.outputs, resp.inputs
    Opinions_no_agent = Opinions.T

    #Initial guess agent simulation
    resp = ct.input_output_response(
                            sys, timepts, U0_initial_guess, x0,
                            t_eval= t_eval, params = params)

    T, Opinions, U = resp.time, resp.outputs, resp.inputs
    Opinions_initial_guess = Opinions.T
    U0_initial_guess_eval = U


    #calculate objective with no agent
    ts = np.arange(0,npts_eval,1)
    ind = np.concatenate((ts[::int(npts_eval/(npts-1))], [ts[-1]]))

    coeffs_no_agent = np.ones(npts*(ne+nv))
    coeffs_no_agent[npts*ne:] = Opinions_no_agent.T[:,ind].reshape(npts*nv,1).T
    obj_no_agent = ocp._cost_function(coeffs_no_agent)

    #calculate objective with initial guess.  also set initial guess for ocp.initial_guess using accurate opinions from simulator
    ocp.initial_guess[npts*ne:] = Opinions_initial_guess.T[:,ind].reshape(npts*nv,1).T
    obj_initial_guess = ocp._cost_function(ocp.initial_guess)
    
    print(f"No shadowbanning: {integral_cost.__name__} = {obj_no_agent:.3f}")
    print(f"Initial shadowbanning guess: {integral_cost.__name__} = {obj_initial_guess:.3f}")
    ####################
    #pyoptsparse functions
    def objfunc(xdict):
        x = np.vstack([xdict[f"x{k}"] for k in range(npts)]).T.reshape(nv*npts)
        u = np.vstack([xdict[f"u{k}"] for k in range(npts)]).T.reshape(ne*npts)
        coeffs = np.concatenate((u, x))

        funcs = {}
        funcs["cost"] = ocp._cost_function(coeffs)
        for k in range(npts):
            if k ==0:
                funcs[f"cons_collocation_0"] = ocp._collocation_constraint(coeffs)[k::npts] +ocp.x 
            else:
                funcs[f"cons_collocation_{k}"] = ocp._collocation_constraint(coeffs)[k::npts]       

        return funcs, False

    def sens_cons(xdict, dt, nv, npts):
        I_nv = sparse.identity(nv)
          
        J={}
        for k in range(1,npts):  #dont need k=0 because that is a linear constraint on the initial opinions
                Jk = {}
                for l in [k-1,k]:
                    Jk[f'u{k-1}'] = -dt/2*jac_cons_collocation_u(xdict[f"x{k-1}"], xdict[f"u{k-1}"])
                    Jk[f'u{k}']   = -dt/2*jac_cons_collocation_u(xdict[f"x{k}"],   xdict[f"u{k}"])  

                    Jk[f'x{k-1}'] = -I_nv -dt/2*jac_cons_collocation_x(xdict[f"x{k-1}"], xdict[f"u{k-1}"])
                    Jk[f'x{k}']   =  I_nv -dt/2*jac_cons_collocation_x(xdict[f"x{k}"],   xdict[f"u{k}"])
                J[f"cons_collocation_{k}"] = Jk

        return J
          
    def sens_mean(xdict, funcs_dict):   
        nv = len(xdict['x0'])
        ne = len(xdict['u0'])
        npts = int(len(xdict)/2)
        J = sens_cons(xdict, dt, nv, npts)
        Jcost = {}
        for k in range(npts):
            fx = -1/(npts-1)/nv*np.ones(nv)
            fu = -alpha/(npts-1)/ne*np.ones(ne) 
            if k ==0 or k==npts-1:
                Jcost[f"x{k}"] = fx/2
                Jcost[f"u{k}"] = fu/2
            else:
                Jcost[f"x{k}"] = fx
                Jcost[f"u{k}"] = fu
        J['cost'] = Jcost


        return J, False
          
    def sens_varmax(xdict, funcs_dict):  
        '''Jacobian sens function when using varmax as integral cost'''
        nv = len(xdict['x0'])
        ne = len(xdict['u0'])
        npts = int(len(xdict)/2)

        J = sens_cons(xdict, dt, nv, npts)
        
        Jcost = {}
        for k in range(npts):
            mu = np.mean(xdict[f"x{k}"])
            fx = -1/(npts-1)/(nv-1)*2*(1-1/nv)*(xdict[f"x{k}"] -mu)
            fu = -alpha/(npts-1)/ne*np.ones(ne) 
            if k ==0 or k==npts-1:
                Jcost[f"x{k}"] = fx/2
                Jcost[f"u{k}"] = fu/2
            else:
                Jcost[f"x{k}"] = fx
                Jcost[f"u{k}"] = fu
        J['cost'] = Jcost


        return J, False
          
    def sens_varmin(xdict, funcs_dict):   
        nv = len(xdict['x0'])
        ne = len(xdict['u0'])
        npts = int(len(xdict)/2)

        J = sens_cons(xdict, dt, nv, npts)
        Jcost = {}
        for k in range(npts):
            mu = np.mean(xdict[f"x{k}"])
            fx = 1/(npts-1)/(nv-1)*2*(1-1/nv)*(xdict[f"x{k}"] -mu)
            fu = -alpha/(npts-1)/ne*np.ones(ne) 
            if k ==0 or k==npts-1:
                Jcost[f"x{k}"] = fx/2
                Jcost[f"u{k}"] = fu/2
            else:
                Jcost[f"x{k}"] = fx
                Jcost[f"u{k}"] = fu
        J['cost'] = Jcost

        return J, False
          
    def jac_cons_collocation_u(x,u):
        data = shift(x[A.row]- x[A.col],tau,omega) #shift value
        Shift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
        Rate_matrix = diags(params['rates'],0) #create a diagonal matrix with Rates values
        jac0 = Rate_matrix @ Shift_matrix # matrix multiply

        s = jac0[A.row, A.col].A[0]

        #print(f"len(jac0.data) = {len(jac0.data)}, len(s) = {len(s)} len(E.row) = {len(E.row)}, len(E.col) = {len(E.col)}")
        jac = coo_matrix((s, (E.row, E.col)), shape=E.shape) #create shift matrix in coordinate format (row index, col index, value)

        return jac.toarray()

    def jac_cons_collocation_x(x,u):
        data = dshift(x[A.row]- x[A.col],tau,omega) #shift value
        dShift_matrix = coo_matrix((data, (A.row, A.col)), shape=A.shape) #create shift matrix in coordinate format (row index, col index, value)
        U = coo_matrix((u, (A.row, A.col)), shape=A.shape)
        dShift_matrix = dShift_matrix.multiply(U)
        Rate_matrix = diags(params['rates'],0) #create a diagonal matrix with Rates values

        jac0 = Rate_matrix @ dShift_matrix # matrix multiply
        Dsum = jac0.sum(axis = 0).A1 #contribution from following of node
        # Assign Dsum to the diagonal of jac
        jac_diag  = diags(Dsum,0)
        jac = -jac_diag +  jac0.T 


        return jac.toarray()
    #############################
    #create pyoptsparse problem
    problem = Optimization(f"{nv} node shadowban network", objfunc)   
    for k in range(npts):
        problem.addVarGroup(f"u{k}", ne, "c", lower = 0, upper = 1, value = ocp.initial_guess[k:npts*ne:npts] )
        problem.addVarGroup(f"x{k}", nv, "c", lower = 0, upper = 1, value =  ocp.initial_guess[npts*ne+k::npts])
        if k == 0:
            row, col, data = np.arange(nv), np.arange(nv),np.ones(nv)
            I_nv = {'coo':[row,  col,    data], 'shape':[nv, nv]} # A coo matrix
            problem.addConGroup(f"cons_collocation_0", nv, lower = ocp.x , upper = ocp.x, wrt = ['x0'], linear = True, jac={"x0": I_nv})

        else:
            problem.addConGroup(f"cons_collocation_{k}", nv, lower = 0, upper = 0, wrt = [f'x{k-1}', f"x{k}",f'u{k-1}', f"u{k}"])


    #objective
    problem.addObj('cost')
    problem.printSparsity()
  
    #####################################
    #sens function
    if OBJECTIVE == 'MEAN':
        sens = sens_mean
    elif  OBJECTIVE == 'VARMAX':
        sens = sens_varmax
    elif  OBJECTIVE == 'VARMIN':
        sens = sens_varmin
    else:
        print("Error: Wrong Objective.  Choose from NONE, MEAN, VARMAX, VARMIN")
        sens = -1
    #############################
    #create pyoptsparse optimizer

    
    #solve problem
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"{current_time}\nSparsity and jacobian ipopt for {nv} nodes")
    sol = optimizer(problem, sens= sens)
          
    # Play beep sound when code execution is done
    frequency = 2500  # Set the beep frequency (in Hz)
    duration = 1000  # Set the beep duration (in milliseconds)
    winsound.Beep(frequency, duration)
    output = {}
    output['obj_no_agent'] = obj_no_agent
    output['obj_initial_guess'] = obj_initial_guess
    output['sol'] = sol
    output['sys'] = sys
    output['problem'] = problem
    output['ocp'] = ocp
    
    print(f"Optimal shadowbanning: {integral_cost.__name__} = {sol.fStar[0]:.3f}")

    return output





