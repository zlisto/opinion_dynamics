import numpy as np
from typing import List, Set, Dict, Tuple
import pandas as pd
import networkx as nx
import scipy
import math
import matplotlib.pyplot as plt
import scripts.opinion_simulator as sim


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def dsigmoid(x):
  y = sigmoid(x)
  return y*(1-y)

# ADAM step
def ADAM_step(gradient, M_old, V_old, t, beta1=0.9, beta2=0.999,  epsilon=1e-8):
  nparam = len(gradient)
  M = np.zeros(nparam)
  V = np.zeros(nparam)
  step = np.zeros(nparam)
  for i in range(nparam):
    m_old = M_old[i]
    v_old = V_old[i]
    m = beta1*m_old + (1.0-beta1)*gradient[i]
    v = beta2*v_old + (1.0-beta2)*gradient[i]**2
    m_unbias = m/(1-beta1**(t+1))
    v_unbias = v/(1-beta2**(t+1))
    step[i] = m_unbias/(np.sqrt(v_unbias)+epsilon)
    M[i] = m
    V[i] = v

  return (step,M,V)

def bang_bang_two_slope(t, u0,slope1, slope2, t_switch):
    h = sigmoid((t-np.exp(t_switch))*10)
    agent_opinions = u0 - np.exp(slope1)*t + h*(np.exp(slope1)+np.exp(slope2))*(t-np.exp(t_switch))
    #agent_opinions[agent_opinions<0] = 0
    #agent_opinions[agent_opinions>1] = 1
    agent_opinions = np.reshape(agent_opinions,(len(t),1))
    return agent_opinions

def optimize_agent(opinions0,rates,A,
                  params_initial,agents_rates,targets_indices,OBJECTIVE,
                   nsteps,tstep,
                   OPT='ADAM',grad_step=0.01, params_steps = [0.1,0.1,0.1,0.1],iter_max=1000,
                   plots = True, plots_freq = 50):
    print(f"Optimizer = {OPT}\nObjective = {OBJECTIVE}")
    u0 = params_initial[0]
    slope1 = params_initial[1]
    slope2 = params_initial[2]
    tswitch = params_initial[3]
    
    u0_step = params_steps[0]
    slope1_step = params_steps[1]
    slope2_step = params_steps[2]
    tswitch_step = params_steps[3]

    tmax = nsteps*tstep
    T = np.linspace(0, tmax, num=nsteps)

    assert len(T) == nsteps
    agents_opinions = bang_bang_two_slope(T, u0,slope1, slope2, tswitch)

    params = np.zeros((iter_max,len(params_initial)))
    Js = np.zeros((iter_max,1))
    for iteration in range(iter_max):
        params[iteration,:] = [u0,slope1,slope2,tswitch]
        (Opinions,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions, 
                                             agents_rates, targets_indices, nsteps, tstep)
        J = sim.objective_value(OBJECTIVE,Opinions)
        Js[iteration,:] = J
        if iteration%plots_freq==0:
            print(f"Iteration {iteration}: Objective = {J}")
            if plots == True:
              plt.plot(T,Opinions)
              plt.plot(T,agents_opinions, label = 'agent', marker = '.')
              plt.legend()
              plt.title(f"Iteration {iteration}: Obj = {J}")
              plt.grid()
              plt.show()

        #positive parameter shift  
        agents_opinions_u0 = bang_bang_two_slope(T, u0+u0_step,slope1, slope2, tswitch)
        agents_opinions_slope1 = bang_bang_two_slope(T, u0,slope1+slope1_step, slope2, tswitch)
        agents_opinions_slope2 = bang_bang_two_slope(T, u0,slope1, slope2+slope2_step, tswitch)
        agents_opinions_tswitch = bang_bang_two_slope(T, u0,slope1, slope2, tswitch+tswitch_step)

        #negative parameter shift
        agents_opinions_u0_neg = bang_bang_two_slope(T, u0-u0_step,slope1, slope2, tswitch)
        agents_opinions_slope1_neg = bang_bang_two_slope(T, u0,slope1-slope1_step, slope2, tswitch)
        agents_opinions_slope2_neg = bang_bang_two_slope(T, u0,slope1, slope2-slope2_step, tswitch)
        agents_opinions_tswitch_neg = bang_bang_two_slope(T, u0,slope1, slope2, tswitch-tswitch_step)

        #no parameter shift
        (Opinions,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions, 
                                             agents_rates, targets_indices, nsteps, tstep)
        #positive parameter shift
        (Opinions_u0,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_u0, 
                                             agents_rates, targets_indices, nsteps, tstep)
        (Opinions_slope1,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_slope1, 
                                             agents_rates, targets_indices, nsteps, tstep)
        (Opinions_slope2,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_slope2, 
                                             agents_rates, targets_indices, nsteps, tstep)
        (Opinions_tswitch,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_tswitch, 
                                             agents_rates, targets_indices, nsteps, tstep)
        #negative parameter shift
        (Opinions_u0_neg,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_u0_neg, 
                                             agents_rates, targets_indices, nsteps, tstep)

        (Opinions_slope1_neg,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_slope1_neg, 
                                             agents_rates, targets_indices, nsteps, tstep)
        (Opinions_slope2_neg,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_slope2_neg, 
                                             agents_rates, targets_indices, nsteps, tstep)
        (Opinions_tswitch_neg,DxDt) = sim.simulate_opinion(opinions0, rates, A, 
                                              agents_opinions_tswitch_neg, 
                                             agents_rates, targets_indices, nsteps, tstep)

        Ju0 = sim.objective_value(OBJECTIVE,Opinions_u0)
        Jslope1 = sim.objective_value(OBJECTIVE,Opinions_slope1)
        Jslope2 = sim.objective_value(OBJECTIVE,Opinions_slope2)
        Jtswitch = sim.objective_value(OBJECTIVE,Opinions_tswitch)
        #negative parameter shift
        Ju0_neg = sim.objective_value(OBJECTIVE,Opinions_u0_neg)
        Jslope1_neg = sim.objective_value(OBJECTIVE,Opinions_slope1_neg)
        Jslope2_neg = sim.objective_value(OBJECTIVE,Opinions_slope2_neg)
        Jtswitch_neg = sim.objective_value(OBJECTIVE,Opinions_tswitch_neg)   

        #calculate gradient of objective wrt parameters of policy
        gu0 = (Ju0-Ju0_neg)/u0_step
        gslope1 = (Jslope1-Jslope1_neg)/slope1_step/2
        gslope2 = (Jslope2-Jslope2_neg)/slope2_step/2       
        gtswitch = (Jtswitch-Jtswitch_neg)/tswitch_step/2

        gradient = [gu0,gslope1,gslope2,gtswitch]
        #parameter update
        if OPT == 'GRAD':
            u0 -= gradient[0]*grad_step
            slope1 -= gradient[1]*grad_step
            slope2 -= gradient[2]*grad_step    
            tswitch -= gradient[3]*grad_step 
        elif OPT =='ADAM':
            if iteration ==0:
                M_old = np.zeros(len(gradient))
                V_old = np.zeros(len(gradient))
            else:
                M_old = M
                V_old = V
            (adam_step,M,V) = ADAM_step(gradient, M_old, V_old, iteration)
            u0 -= adam_step[0]*grad_step
            slope1 -= adam_step[1]*grad_step
            slope2 -= adam_step[2]*grad_step    
            tswitch -= adam_step[3]*grad_step 

        agents_opinions = bang_bang_two_slope(T, u0,slope1, slope2, tswitch)
    return (Js,params)