import numpy as np
import scipy.sparse as sparse
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse import coo_matrix, csr_matrix, diags, identity

from scipy.optimize import linprog
import networkx as nx
import scripts.shadowban_pyoptsparse as sb




def sys_update_mean(t, x, u, params):
    # Get the parameters for the model
    A = params.get('A')         # vehicle wheelbase
    rates = params.get('rates')
    tau = params.get('tau')
    omega = params.get('omega')
    #smart initial condition (for maximizing mean)
    data = sb.shift(x[A.row]- x[A.col],tau,omega)
    u_smart = (data>0).astype(int)
    
    # Create COO matrix U with data from u at A's locations
    
    U_row = A.row
    U_col = A.col
    U = coo_matrix((u_smart, (U_row, U_col)), shape=(A.shape[0], A.shape[1]))
    #print(x)
    # Return the derivative of the state
    return sb.dxdt(x, t, rates, A, tau, omega, U)

def sys_update_varmin(t, x, u, params):
    # Get the parameters for the model
    A = params.get('A')         # vehicle wheelbase
    rates = params.get('rates')
    tau = params.get('tau')
    omega = params.get('omega')

    #smart initial condition (for maximizing mean)
    data = sb.shift(x[A.row]- x[A.col],tau,omega)
    u_smart = (np.sign(x[A.col]-x.mean())*np.sign(data)<=0).astype(int)
    
    # Create COO matrix U with data from u at A's locations
    
    U_row = A.row
    U_col = A.col
    U = coo_matrix((u_smart, (U_row, U_col)), shape=(A.shape[0], A.shape[1]))
    #print(x)
    # Return the derivative of the state
    return sb.dxdt(x, t, rates, A, tau, omega, U)

def sys_update_varmax(t, x, u, params):
    # Get the parameters for the model
    A = params.get('A')         # vehicle wheelbase
    rates = params.get('rates')
    tau = params.get('tau')
    omega = params.get('omega')

    #smart initial condition (for maximizing mean)
    data = sb.shift(x[A.row]- x[A.col],tau,omega)
    u_smart = (np.sign(x[A.col]-x.mean())*np.sign(data)>0).astype(int)
    
    # Create COO matrix U with data from u at A's locations
    
    U_row = A.row
    U_col = A.col
    U = coo_matrix((u_smart, (U_row, U_col)), shape=(A.shape[0], A.shape[1]))
    #print(x)
    # Return the derivative of the state
    return sb.dxdt(x, t, rates, A, tau, omega, U)

def sys_update_lp(t, x, u, params):
    # Get the parameters for the model
    A = params.get('A')         # vehicle wheelbase
    rates = params.get('rates')
    tau = params.get('tau')
    omega = params.get('omega')

    u_smart = shadowban_lp(params,x)
    
    # Create COO matrix U with data from u at A's locations
    
    U_row = A.row
    U_col = A.col
    U = coo_matrix((u_smart, (U_row, U_col)), shape=(A.shape[0], A.shape[1]))
    #print(x)
    # Return the derivative of the state
    return sb.dxdt(x, t, rates, A, tau, omega, U)

def shadowban_lp(params,x):
    E = params['E']
    ne = E.shape[1]
    smax = params['smax']
    c = get_B(params,x)
    A_ub = -np.ones((1,ne))
    b = -ne*(1-smax)
    res = linprog(c, A_ub=A_ub, b_ub=b, bounds=[0,1])
    ustar =res.x
    return ustar

def get_B(params,x):
    rates = params.get('rates')
    A = params.get("A")
    tau = params.get("tau")
    omega = params.get("omega")
    OBJECTIVE = params['OBJECTIVE']
    nv = len(x)
    if OBJECTIVE == 'MEAN':
        C = -1/nv*np.ones(nv)
    elif OBJECTIVE == 'VARMIN':
        mu = np.mean(x)
        C = 2/nv*(x-mu)
    elif OBJECTIVE == 'VARMAX':
        mu = np.mean(x)
        C = -2/nv*(x-mu)
    B = rates[A.row]*sb.shift(x[A.row]-x[A.col], tau, omega)*C[A.col]
    return B

def shadowban_strength(Opinions, params):
    A = params.get('A')         # vehicle wheelbase
    rates = params.get('rates')
    tau = params.get('tau')
    omega = params.get('omega')
    OBJECTIVE = params.get('OBJECTIVE')
    assert OBJECTIVE in ['MEAN', 'VARMIN', 'VARMAX']
    
    Strength = []
    for i in range(Opinions.shape[0]):
        x = Opinions[i, :]
        #smart initial condition (for maximizing mean)
        data = sb.shift(x[A.row]- x[A.col],tau,omega)
        
        if OBJECTIVE == "MEAN":
            u_smart = (data>0).astype(int)
        elif OBJECTIVE == 'VARMIN':
            u_smart = (np.sign(x[A.col]-x.mean())*np.sign(data)<=0).astype(int)
        elif OBJECTIVE == 'VARMAX':
            u_smart = (np.sign(x[A.col]-x.mean())*np.sign(data)>0).astype(int)
        
        Strength.append(1-u_smart.mean())
    Strength = np.array(Strength)
    
    return Strength