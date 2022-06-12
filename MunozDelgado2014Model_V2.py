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
model.TR = pyo.Set(initialize=TR) #Set of Transformers Types
model.K_p = pyo.Set(initialize=[1,2]) #Set of DG options
def K_l_rule(model, l):
    return K_l[l]
model.K_l = pyo.Set(model.L, initialize=K_l_rule) #Set of lines options by type
def K_tr_rule(model, tr):
    return K_tr[tr]
model.K_tr = pyo.Set(model.TR, initialize=K_tr_rule) #Set of new transformers options
model.Omega_SS = pyo.Set(initialize=Omega_SS) #Set of substation nodes
model.Omega_N = pyo.Set(initialize=Omega_N) #Set of all nodes
def Omega_l_s_rule(model, l, s):
    return Omega_l_s[l][s-1]
model.Omega_l_s = pyo.Set(model.L, model.Omega_N, initialize=Omega_l_s_rule) #Sets of nodes connected to node by a feeder of type

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
model.RR_NT = pyo.Param(initialize=RR_NT, domain=Reals) #Capital recovery rates for investment in new transformers.
def RR_p_rule(model, p):
    return RR_p[p]
model.RR_p = pyo.Param(model.P, initialize=RR_p_rule, domain=Reals) #Capital recovery rates for investment in generators.
def C_Il_k_rule(model, typ, l):
    return C_Il_k[typ][l-1]
model.C_Il_k = pyo.Param(model.L_l, model.K_l['NRF'] | model.K_l['NAF'], initialize=C_Il_k_rule) #Investment cost coefficients of feeders.           
def C_ISS_s_rule(model, ss):
    return C_ISS_s[ss] 
model.C_ISS_s = pyo.Param(model.Omega_SS, initialize=C_ISS_s_rule) #Investment cost coefficients of substations.
def C_INT_k_rule(model, nt):
    return C_INT_k[nt-1]
model.C_INT_k = pyo.Param(model.K_tr['NT'], initialize=C_INT_k_rule) #Investment cost coefficients of new transformers.
def C_Ip_k_rule(model, p, k):
    return C_Ip_k[p][k-1]
model.C_Ip_k = pyo.Param(model.P, model.K_p, initialize=C_Ip_k_rule) #Investment cost coefficients of generators.
def l_sr_rule(model, s, r):
    return l__sr[s-1,r-1]
model.l__sr = pyo.Param(model.Omega_N, model.Omega_N, initialize=l_sr_rule) #Feeder length.
model.pf = pyo.Param(initialize=pf, domain=Reals) #System power factor.
def Gup_p_k_rule(model, p, k):
    return Gup_p_k[p][k-1]
model.Gup_p_k = pyo.Param(model.P, model.K_p, initialize=Gup_p_k_rule) #Rated capacities of generators


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

def x_l_rule(m): 
    index = []
    for l in model.L_l:
        for s in model.Omega_N:
            for r in model.Omega_l_s[l,s]:
                for k in model.K_l[l]:
                    for t in model.T:
                        index.append((l,s,r,k,t))
    return index
model.x_l_rule = pyo.Set(dimen=5, initialize=x_l_rule)
model.x_l_srkt = pyo.Var(model.x_l_rule,
                         within=pyo.Binary
    ) #Binary investment variables for feeders.

def x_SS_rule(model):
    index = []
    for s in model.Omega_SS:
        for t in model.T:
            index.append((s,t))
    return index
model.x_SS_rule = pyo.Set(dimen=2, initialize=x_SS_rule)
model.x_SS_st = pyo.Var(model.x_SS_rule,
                        within=pyo.Binary
                        ) #Binary investment variables for substations.

def x_NT_rule(model):
    index = []
    for s in model.Omega_SS:
        for k in model.K_tr['NT']:
            for t in model.T:
                index.append((s,k,t))
    return index
model.x_NT_rule = pyo.Set(dimen=3, initialize=x_NT_rule)
model.x_NT_skt = pyo.Var(model.x_NT_rule, 
                         within=pyo.Binary
    ) #Binary investment variables for new transformers.

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



















