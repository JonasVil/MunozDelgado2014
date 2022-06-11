"""
Implementation on Pyomo of Distribution Expansion Planning Model proposed by MuÃ±oz-Delgado et al. (2014).

Reference:
MuÃ±oz-Delgado, G., Contreras, J., & Arroyo, J. M. (2014). Joint expansion planning of distributed generation and distribution networks. IEEE Transactions on Power Systems, 30(5), 2579-2590.
DOI: 10.1109/TPWRS.2014.2364960

@Code Athor: Jonas Villela de Souza
@Initial Date: June 6, 2022
@Version Date: June 6, 2022
"""

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from Data_24Bus import * 

# =============================================================================
# DG Penetration
# =============================================================================

Vare = 0 #Penetration limit for distributed generation.

# =============================================================================
# Model
# =============================================================================

model = pyo.ConcreteModel()


# =============================================================================
# Sets
# =============================================================================


model.T = pyo.Set(initialize=T) #Set of time stages.
model.L = pyo.Set(initialize=L) #Set of feeder types.
model.P = pyo.Set(initialize=P) #Set of Generator Types

# =============================================================================
# Parameters
# =============================================================================

model.i = pyo.Var(initialize=i) #Annual interest rate.
def RR_l_rule(model, l):
    if l in ['NRF','NAF']:
        index = RR_l[l]
        return index
model.RR_l = pyo.Var(model.L, initialize=RR_l_rule) #Capital recovery rates for investment in feeders.   
model.RR_SS = pyo.Var(initialize=RR_SS) #Capital recovery rates for investment in substations.
model.RR_NT = pyo.Var(initialize=RR_NT) #Capital recovery rates for investment in new transformers
def RR_p_rule(model, p):
    


# =============================================================================
# Variables
# =============================================================================

model.C_I_t = pyo.Var(model.T, 
                      bounds=(0.0,None)
                      )

model.C_M_t = pyo.Var(model.T, 
                      bounds=(0.0,None)
                      )

model.C_E_t = pyo.Var(model.T, 
                      bounds=(0.0,None)
                      )

model.C_R_t = pyo.Var(model.T, 
                      bounds=(0.0,None)
                      )

model.C_U_t = pyo.Var(model.T, 
                      bounds=(0.0,None)
                      )

model.C_TPV = pyo.Var(bounds=(0.0,None)
                      )



# =============================================================================
# Objective Function
# =============================================================================

model.Obj = pyo.Objective(expr=model.C_TPV, sense=pyo.minimize)

# =============================================================================
# Costs Constraints
# =============================================================================

def C_TPV_rule(model):
    return model.C_TPV == (sum(model.C_I_t[t]*(((1+model.i())**(-t))/model.i()) 
                               for t in model.T)
                           + sum((model.C_M_t[t] + model.C_E_t[t] + model.C_R_t[t] + model.C_U_t[t])*((1+model.i())**(-t)) 
                               for t in model.T)
                           + ((model.C_M_t[model.T.at(3)] + model.C_E_t[model.T.at(3)] + model.C_R_t[model.T.at(3)] + model.C_U_t[model.T.at(3)])*((1+model.i())**(-model.T.at(3))/model.i()))
                           ) 
model.eq1 = pyo.Constraint(rule=C_TPV_rule)

# =============================================================================
# def eq2_rule(model,t):
#     return model.C_I_t[t] == (sum(RR_l[l]*sum(sum(C_Il_k[l][k-1]*l__sr[s-1,r-1]*model.x_l_srkt[l,s,r,k,t] 
#                                             for s,r in Upsilon_l[l])
#                                         for k in K_l[l])
#                             for l in ["NRF", "NAF"])
#                             
#                             + RR_SS*sum(C_ISS_s[s]*model.x_SS_st[s,t] 
#                                         for s in Omega_SS) 
#                               
#                             + RR_NT*sum(sum(C_INT_k[k-1]*model.x_NT_skt[s,k,t]
#                                     for s in Omega_SS)
#                                 for k in K_tr["NT"]) 
#                             
#                             + sum(RR_p[p]*sum(sum(C_Ip_k[p][k-1]*pf*Gup_p_k[p][k-1]*model.x_p_skt[p,s,k,t]
#                                         for s in Omega_p[p])
#                                     for k in K_p[p])
#                                 for p in P)                              
#                             )  
# 
# 
# =============================================================================



# =============================================================================
# 
# model.A = Set(initialize=['Scones', 'Tea'])
# lb = {'Scones':2, 'Tea':4}
# ub = {'Scones':5, 'Tea':7}
# def fb(model, i):
#    return (lb[i], ub[i])
# model.PriceToCharge = Var(model.A, domain=PositiveIntegers, bounds=fb)
# 
# =============================================================================





















