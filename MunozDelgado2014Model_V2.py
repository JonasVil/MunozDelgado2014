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
model.L_l = pyo.Set(initialize=['NRF','NAF']) #Set of feeder types.
model.P = pyo.Set(initialize=P) #Set of Generator Types
model.K_p = pyo.Set(initialize=[1,2]) #Set of DG options
model.K_l = pyo.Set(initialize=[1,2]) #Set of lines options
model.K_nt = pyo.Set(initialize=[1,2]) #Set of new transformers options
model.Omega_SS = pyo.Set(initialize=Omega_SS) #Set of substation nodes
model.Omega_N = pyo.Set(initialize=Omega_N) #Set of all nodes
# =============================================================================
# Parameters
# =============================================================================

model.i = pyo.Param(initialize=i, domain=Reals) #Annual interest rate.
def RR_l_rule(model, l):
    if l in ['NRF','NAF']:
        index = RR_l[l]
    else:
        index = 0
    return index
model.RR_l = pyo.Param(model.L, initialize=RR_l_rule, domain=Reals) #Capital recovery rates for investment in feeders.   
model.RR_SS = pyo.Param(initialize=RR_SS, domain=Reals) #Capital recovery rates for investment in substations.
model.RR_NT = pyo.Param(initialize=RR_NT, domain=Reals) #Capital recovery rates for investment in new transformers
def RR_p_rule(model, p):
    return RR_p[p]
model.RR_p = pyo.Param(model.P, initialize=RR_p_rule, domain=Reals) #Capital recovery rates for investment in generators
def C_Il_k_rule(model, typ, l):
    return C_Il_k[typ][l-1]
model.C_Il_k = pyo.Param(model.L_l, model.K_l, initialize=C_Il_k_rule) #Investment cost coefficients of feeders           
def C_ISS_s_rule(model, ss):
    return C_ISS_s[ss] 
model.C_ISS_s = pyo.Param(model.Omega_SS, initialize=C_ISS_s_rule) #Investment cost coefficients of substations
def C_INT_k_rule(model, nt):
    return C_INT_k[nt-1]
model.C_INT_k = pyo.Param(model.K_nt, initialize=C_INT_k_rule) #Investment cost coefficients of new transformers
def C_Ip_k_rule(model, p, k):
    return C_Ip_k[p][k-1]
model.C_Ip_k = pyo.Param(model.P, model.K_p, initialize=C_Ip_k_rule) #Investment cost coefficients of generators
def l_sr_rule(model, s, r):
    return 
   
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



















