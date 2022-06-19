"""
Implementation on Pyomo of Distribution Expansion Planning Model proposed by MuÃ±oz-Delgado et al. (2014).

Reference:
MuÃ±oz-Delgado, G., Contreras, J., & Arroyo, J. M. (2014). Joint expansion planning of distributed generation and distribution networks. IEEE Transactions on Power Systems, 30(5), 2579-2590.
DOI: 10.1109/TPWRS.2014.2364960

@Code Athor: Jonas Villela de Souza
@Initial Date: June 6, 2022
@Version Date: June 17, 2022
"""

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from Data_24Bus_V2 import * 

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
model.B = pyo.Set(initialize=B) #Set of load levels.
model.L = pyo.Set(initialize=L) #Set of feeder types.
model.L_nl = pyo.Set(initialize=['NRF','NAF']) #Set of feeder types.
model.P = pyo.Set(initialize=P) #Set of Generator Types
model.TR = pyo.Set(initialize=TR) #Set of Transformers Types
model.n__V = pyo.Set(initialize=[1, 2, 3]) #Number of blocks of the piecewise linear energy losses.


def K_p_rule(model, p):
    return K_p[p]
model.K_p = pyo.Set(model.P, initialize=K_p_rule) #Set of DG options

def K_l_rule(model, l):
    return K_l[l]
model.K_l = pyo.Set(model.L, initialize=K_l_rule) #Set of lines options by type

def K_tr_rule(model, tr):
    return K_tr[tr]
model.K_tr = pyo.Set(model.TR, initialize=K_tr_rule) #Set of new transformers options
model.Omega_SS = pyo.Set(initialize=Omega_SS) #Set of substation nodes
model.Omega_N = pyo.Set(initialize=Omega_N) #Set of all nodes


def Omega_l_s_rule(model,l,s):   
    return Omega_l_s[l][s-1]
model.Omega_l_s = pyo.Set(model.L, model.Omega_N | model.Omega_SS, initialize=Omega_l_s_rule) #Sets of nodes connected to node by a feeder of type

# =============================================================================
# def Omega_l_s_rule(model,l,s):   
#     if len(Omega_l_s[l][s-1]) != 0:
#         return Omega_l_s[l][s-1]
#     else:
#         return pyo.Set.Skip
# model.Omega_l_s = pyo.Set(model.L, model.Omega_N | model.Omega_SS, initialize=Omega_l_s_rule) #Sets of nodes connected to node by a feeder of type
# =============================================================================

def Omega_p_rule(model, p):
    return Omega_p[p]
model.Omega_p = pyo.Set(model.P, initialize=Omega_p_rule) #Sets of possible nodes to install DGs by type

def Upsilon_l_rule(model,l):
    return Upsilon_l[l]
model.Upsilon_l = pyo.Set(model.L, initialize=Upsilon_l_rule) #Set of branches with feeders of type l.

def Upsilon_N_rule(model,l):
    return Upsilon_N[l]
model.Upsilon_N = pyo.Set(model.L, initialize=Upsilon_N_rule) #Set of nodes with feeders of type l.

def Omega_LN_t_rule(model, t):
    return Omega_LN_t[t]
model.Omega_LN_t = pyo.Param(model.T, initialize=Omega_LN_t_rule)

# =============================================================================
# Parameters
# =============================================================================

model.H = pyo.Param(initialize=H, domain=Reals)

model.Vare = pyo.Param(initialize=Vare, domain=Reals) #Penetration limit for distributed generation.

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
model.C_Il_k = pyo.Param(model.L_nl, model.K_l['NRF'] | model.K_l['NAF'], initialize=C_Il_k_rule) #Investment cost coefficients of feeders.           

def C_ISS_s_rule(model, ss):
    return C_ISS_s[ss] 
model.C_ISS_s = pyo.Param(model.Omega_SS, initialize=C_ISS_s_rule) #Investment cost coefficients of substations.

def C_INT_k_rule(model, nt):
    return C_INT_k[nt-1]
model.C_INT_k = pyo.Param(model.K_tr['NT'], initialize=C_INT_k_rule) #Investment cost coefficients of new transformers.

def C_Ip_k_rule(model, p, k):
    return C_Ip_k[p][k-1]
model.C_Ip_k = pyo.Param(model.P, model.K_p['C'] | model.K_p['W'], initialize=C_Ip_k_rule) #Investment cost coefficients of generators.

def l_sr_rule(model):
    index = {}
    for (s,r),l,typ in branch:
        index[s,r] = l
    return index
model.l__sr = pyo.Param(model.Omega_N, model.Omega_N | model.Omega_SS, initialize=l_sr_rule) #Feeder length.

model.pf = pyo.Param(initialize=pf, domain=Reals) #System power factor.

def Gup_p_k_rule(model, p, k):
    return Gup_p_k[p][k-1]
model.Gup_p_k = pyo.Param(model.P, model.K_p['C'] | model.K_p['W'], initialize=Gup_p_k_rule) #Rated capacities of generators

def C_Ml_k_rule(model):
    index = {}
    for l in model.L:
        for k in model.K_l[l]:
            index[l,k] = C_Ml_k[l][k-1]
    return index
model.C_Ml_k = pyo.Param(model.L, model.K_l['NAF'], initialize=C_Ml_k_rule) #Maintenance cost coefficients of feeders

def C_Mtr_k_rule(model):
    index = {}
    for tr in model.TR:
        for k in model.K_tr[tr]:
            index[tr,k] = C_Mtr_k[tr][k-1]
    return index
model.C_Mtr_k = pyo.Param(model.TR, model.K_tr['NT'], initialize=C_Mtr_k_rule) #Maintenance cost coefficients of transformers

def C_Mp_k_rule(model, p, k):
    return C_Mp_k[p][k-1]
model.C_Mp_k = pyo.Param(model.P, model.K_p['C'] | model.K_p['W'], initialize=C_Mp_k_rule) #Maintenance cost coefficients of generators

def C_SS_b_rule(model, b):
    return C_SS_b[b-1]
model.C_SS_b = pyo.Param(model.B, initialize=C_SS_b_rule) #the costs of the energy supplied by all substations.

def C_Ep_k_rule(model, p, k):
    return C_Ep_k[p][k-1]
model.C_Ep_k = pyo.Param(model.P, model.K_p['C'], initialize=C_Ep_k_rule) #the costs of the energy supplied by DG units.
#DG units

model.C_U = pyo.Param(initialize=C_U) #Cost for unserved energy.

def Delta__b_rule(model, b):
    return Delta__b[b-1]
model.Delta__b = pyo.Param(model.B, initialize=Delta__b_rule) #Duration of load level b.

def M_tr_kV_rule(model):
    index = {}
    for tr in model.TR:
        for k in model.K_tr[tr]:
            for y in model.n__V:
                index[tr,k,y] = M_tr_kV[tr][k-1][y-1]
    return index
model.M_tr_kV = pyo.Param(model.TR, model.K_tr['NT'], model.n__V, initialize=M_tr_kV_rule) #Slope of block v of the piecewise linear energy losses for transformers.

def M_l_kV_rule(model):
    index = {}
    for l in model.L:
        for k in model.K_l[l]:
            for z in model.n__V:
                index[l,k,z] = M_l_kV[l][k-1][z-1]
    return index
model.M_l_kV = pyo.Param(model.L, model.K_l['NAF'], model.n__V, initialize=M_l_kV_rule) #Slope of block v of the piecewise linear energy losses for feeders.

def A_tr_kV_rule(model):
    index = {}
    for tr in model.TR:
        for k in model.K_tr[tr]:
            for v in model.n__V:
                index[tr,k,v] = A_tr_kV[tr][k-1][v-1]
    return index
model.A_tr_kV = pyo.Param(model.TR, model.K_tr['NT'], model.n__V, initialize=A_tr_kV_rule) #Width of block v of the piecewise linear energy losses for transformers.

def A_l_kV_rule(model):
    index = {}
    for l in model.L:
        for k in model.K_l[l]:
            for v in model.n__V:
                index[l,k,v] = A_l_kV[l][k-1][v-1]
    return index
model.A_l_kV = pyo.Param(model.L, model.K_l['NAF'],model.n__V, initialize=A_l_kV_rule) #Width of block v of the piecewise linear energy losses for feeders.

model.Vbase = pyo.Param(initialize=Vbase) #Base voltage.

model.V_ = pyo.Param(initialize=V_) #Lower bound for nodal voltages.

model.Vup = pyo.Param(initialize=Vup) #Upper bound for nodal voltages.

def Fup_l_k_rule(model):
    index = {}
    for l in model.L:
        for k in model.K_l[l]:
            index[l,k] = Fup_l_k[l][k-1]
    return index
model.Fup_l_k = pyo.Param(model.L, model.K_l['NAF'], initialize=Fup_l_k_rule) #Upper limit for actual current flows through (MVA).

def Gup_tr_k_rule(model):
    index = {}
    for tr in model.TR:
        for k in model.K_tr[tr]:
            index[tr,k] = Gup_tr_k[tr][k-1]
    return index
model.Gup_tr_k = pyo.Param(model.TR, model.K_tr['NT'], initialize=Gup_tr_k_rule) #Upper limit for current injections of transformers.

def Mi__b_rule(model, b):
    return Mi__b[b-1]
model.Mi__b = pyo.Param(model.B, initialize=Mi__b_rule) #Loading factor of load level b.

def D__st_rule(model):
    index = {}
    for s in model.Omega_N:
        for t in model.T:
            index[s,t] = D__st[s-1,t-1]
    return index
model.D__st = pyo.Param(model.Omega_N, model.T, initialize=D__st_rule) #Actual nodal peak demand.

def Gmax_W_sktb_rule(model, s, k, t, b):
    return Gmax_W_sktb[s-1,k-1,t-1,b-1]
model.Gmax_W_sktb = pyo.Param(model.Omega_p["W"], model.K_p["W"], model.T, model.B, initialize=Gmax_W_sktb_rule) #Maximum wind power availability.

def Z_l_k_rule(model):
    index = {}
    for l in model.L:
        for k in model.K_l[l]:
            index[l,k] = Z_l_k[l][k-1]
    return index
model.Z_l_k = pyo.Param(model.L, model.K_l['NAF'], initialize=Z_l_k_rule)

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

model.d_U_stb = pyo.Var(model.Omega_N, 
                        model.T,
                        model.B,
                        bounds=(0.0,None)
                        )

def x_l_rule(m): 
    index = []
    for l in model.L_nl:
        for s,r in model.Upsilon_l[l]:
            for k in model.K_l[l]:
                for t in model.T:
                    index.append((l,s,r,k,t))
                    index.append((l,r,s,k,t))
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

def x_p_rule(model):
    index = []
    for p in model.P:
        for s in model.Omega_p[p]:
            for k in model.K_p[p]:
                for t in model.T:
                    index.append((p,s,k,t))
    return index 

model.x_p_rule = pyo.Set(dimen=4, initialize=x_p_rule)
model.x_p_skt = pyo.Var(model.x_p_rule,
                        within=pyo.Binary
    ) #Binary investment variables for generators.


def y_l_rule(model):
    index = []
    for l in model.L:
        for s,r in model.Upsilon_l[l]:
            for k in model.K_l[l]:
                for t in model.T:
                    index.append((l,s,r,k,t))
                    index.append((l,r,s,k,t))
    return index 

model.y_l_rule = pyo.Set(dimen=5, initialize=y_l_rule)
model.y_l_srkt = pyo.Var(model.y_l_rule,
                         within=pyo.Binary
    )


def y_tr_rule(model):
    index = []
    for tr in model.TR:
        for s in model.Omega_SS:
            for k in model.K_tr[tr]:
                for t in model.T:
                    index.append((tr,s,k,t))
    return index 

model.y_tr_rule = pyo.Set(dimen=4, initialize=y_tr_rule)
model.y_tr_skt = pyo.Var(model.y_tr_rule,
                         within=pyo.Binary
    )

def y_p_rule(model):
    index = []
    for p in model.P:
        for s in model.Omega_N:
            for k in model.K_p[p]:
                for t in model.T:
                    index.append((p,s,k,t))
    return index 

model.y_p_rule = pyo.Set(dimen=4, initialize=y_p_rule)
model.y_p_skt = pyo.Var(model.y_p_rule,
                        within=pyo.Binary
    )

def g_tr_rule(model):
    index = []
    for tr in model.TR:
        for s in model.Omega_SS:
            for k in model.K_tr[tr]:
                for t in model.T:
                    for b in model.B:
                        index.append((tr,s,k,t,b))
    return index

model.g_tr_rule = pyo.Set(dimen=5, initialize=g_tr_rule)
model.g_tr_sktb = pyo.Var(model.g_tr_rule,
                         bounds=(0.0,None)                        
    )

def g_p_rule(model):
    index = []
    for p in model.P:
        for s in model.Omega_p[p]:
            for k in model.K_p[p]:
                for t in model.T:
                    for b in model.B:
                        index.append((p,s,k,t,b))
    return index

model.g_p_rule = pyo.Set(dimen=5, initialize=g_p_rule)
model.g_p_sktb = pyo.Var(model.g_p_rule,
                         bounds=(0.0,None)                    
    )

def delta_tr_rule(m):
    index = []
    for tr in model.TR:
        for s in model.Omega_SS:
            for k in model.K_tr[tr]:
                for t in model.T:
                    for b in model.B:
                        for v in model.n__V:
                            index.append((tr,s,k,t,b,v))
    return index 

model.delta_tr_rule = pyo.Set(dimen=6, initialize=delta_tr_rule)
model.delta_tr_sktbv = pyo.Var(model.delta_tr_rule,
                               bounds=(0.0,None)
    )
           
def delta_l_rule(m):
    index = []
    for l in model.L:
        for s,r in model.Upsilon_l[l]:
            for k in model.K_l[l]:
                for t in model.T:
                    for b in model.B:
                        for v in model.n__V:
                            index.append((l,s,r,k,t,b,v))
                            index.append((l,r,s,k,t,b,v))
    return index 

model.delta_l_rule = pyo.Set(dimen=7, initialize=delta_l_rule)
model.delta_l_srktbv = pyo.Var(model.delta_l_rule,
                               bounds=(0.0,None)                             
    )

def f_l_rule(m):
    index = []
    for l in model.L:
        for s,r in model.Upsilon_l[l]:
            for k in model.K_l[l]:
                for t in model.T:
                    for b in model.B:
                        index.append((l,s,r,k,t,b))
                        index.append((l,r,s,k,t,b))
    return index
            
model.f_l_rule = pyo.Set(dimen=6, initialize=f_l_rule)
model.f_l_srktb = pyo.Var(model.f_l_rule,
                          bounds=(0.0,None)
    )
model.ftio_l_srktb = pyo.Var(model.f_l_rule,
                          bounds=(0.0,None)
    )

model.V_stb = pyo.Var(model.Omega_N | model.Omega_SS, 
                      model.T, 
                      model.B,
                      bounds=(0.0,None)
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

def eq2_rule(model,t):
    return model.C_I_t[t] == (sum(model.RR_l[l]*sum(sum(model.C_Il_k[l,k]*model.l__sr[s,r]*model.x_l_srkt[l,s,r,k,t] 
                                            for s,r in model.Upsilon_l[l])
                                        for k in model.K_l[l])
                            for l in model.L_nl)
                            
                            + model.RR_SS*sum(model.C_ISS_s[s]*model.x_SS_st[s,t] 
                                        for s in model.Omega_SS) 
                              
                            + model.RR_NT*sum(sum(model.C_INT_k[k]*model.x_NT_skt[s,k,t]
                                    for s in model.Omega_SS)
                                for k in model.K_tr['NT']) 
                            
                            + sum(model.RR_p[p]*sum(sum(model.C_Ip_k[p,k]*model.pf*model.Gup_p_k[p,k]*model.x_p_skt[p,s,k,t]
                                        for s in model.Omega_p[p])
                                    for k in model.K_p[p])
                                for p in model.P)                              
                            )  
model.eq2 = pyo.Constraint(model.T, rule=eq2_rule)

def eq3_rule(model,t):
    return model.C_M_t[t] == (sum(sum(sum(model.C_Ml_k[l,k]*(model.y_l_srkt[l,s,r,k,t] + model.y_l_srkt[l,r,s,k,t])
                    for s,r in model.Upsilon_l[l])
                for k in model.K_l[l])
            for l in model.L)
            
            + sum(sum(sum(model.C_Mtr_k[tr,k]*model.y_tr_skt[tr,s,k,t]
                    for s in model.Omega_SS)
                for k in model.K_tr[tr])
            for tr in model.TR)
            
            + sum(sum(sum(model.C_Mp_k[p,k]*model.y_p_skt[p,s,k,t]
                    for s in model.Omega_p[p])
                for k in model.K_p[p])
            for p in model.P)
        
        )
model.eq3 = pyo.Constraint(model.T, rule=eq3_rule)

def eq4_rule(model,t):
    return model.C_E_t[t] == (sum(model.Delta__b[b]*model.pf*(sum(sum(sum(model.C_SS_b[b]*model.g_tr_sktb[tr,s,k,t,b]
                                for s in model.Omega_SS)
                            for k in model.K_tr[tr])
                        for tr in model.TR)
                        
                        + sum(sum(sum(model.C_Ep_k[p,k]*model.g_p_sktb[p,s,k,t,b]
                                for s in model.Omega_p[p])
                            for k in model.K_p[p])
                        for p in model.P)
                        )
                for b in model.B)
            )
model.eq4 = pyo.Constraint(model.T, rule=eq4_rule)

def eq5_rule(model,t):
    return model.C_R_t[t] == (sum(model.Delta__b[b]*model.C_SS_b[b]*model.pf*(sum(sum(sum(sum(
                                model.M_tr_kV[tr,k,y]*model.delta_tr_sktbv[tr,s,k,t,b,y]
                                for y in model.n__V)
                            for s in model.Omega_SS)
                        for k in model.K_tr[tr])
                    for tr in model.TR)

                    + sum(sum(sum(sum(model.M_l_kV[l,k,z]*model.l__sr[s,r]*(model.delta_l_srktbv[l,s,r,k,t,b,z] + model.delta_l_srktbv[l,r,s,k,t,b,z])
                                for z in model.n__V)
                            for s, r in model.Upsilon_l[l])
                        for k in model.K_l[l])
                    for l in model.L)
                )
            for b in model.B)
        )
model.eq5 = pyo.Constraint(model.T, rule=eq5_rule)

model.eq5_aux1 = pyo.ConstraintList()
for tr in model.TR:
    for s in model.Omega_SS:
        for k in model.K_tr[tr]:
            for t in model.T:
                for b in model.B:
                    model.eq5_aux1.add(model.g_tr_sktb[tr,s,k,t,b] == sum(model.delta_tr_sktbv[tr,s,k,t,b,v] for v in model.n__V))

model.eq5_aux2 = pyo.ConstraintList()
for tr in model.TR:
    for s in model.Omega_SS:
        for k in model.K_tr[tr]:
            for t in model.T:
                for b in model.B:
                    for v in model.n__V:
                        model.eq5_aux2.add(model.delta_tr_sktbv[tr,s,k,t,b,v] <= model.A_tr_kV[tr,k,v])


model.eq5_aux3 = pyo.ConstraintList()
for l in model.L:
    for s,r in model.Upsilon_l[l]:
        for k in model.K_l[l]:
            for t in model.T:
                for b in model.B:
                    model.eq5_aux3.add(model.f_l_srktb[l,s,r,k,t,b] == sum(model.delta_l_srktbv[l,s,r,k,t,b,v] for v in model.n__V))
                    model.eq5_aux3.add(model.f_l_srktb[l,r,s,k,t,b] == sum(model.delta_l_srktbv[l,r,s,k,t,b,v] for v in model.n__V))

model.eq5_aux4 = pyo.ConstraintList()
for l in model.L:
    for s,r in model.Upsilon_l[l]:
        for k in model.K_l[l]:
            for t in model.T:
                for b in model.B:
                    for v in model.n__V:
                        model.eq5_aux4.add(model.delta_l_srktbv[l,s,r,k,t,b,v] <= model.A_l_kV[l,k,v])
                        model.eq5_aux4.add(model.delta_l_srktbv[l,r,s,k,t,b,v] <= model.A_l_kV[l,k,v])


def eq6_rule(model,t):
    return model.C_U_t[t] == (sum(sum(model.Delta__b[b]*model.C_U*model.pf*model.d_U_stb[s,t,b]
                for s in model.Omega_LN_t[t])
            for b in model.B)
        )
model.eq6 = pyo.Constraint(model.T, rule=eq6_rule)

# =============================================================================
# Kirchhoff's Laws and Operational Limits
# =============================================================================

def eq7_rule(model, s, t, b):
    return pyo.inequality(model.V_ ,model.V_stb[s,t,b], model.Vup)
model.eq7 = pyo.Constraint(model.Omega_N | model.Omega_SS, model.T, model.B, rule=eq7_rule)

model.eq8 = pyo.ConstraintList()
for l in model.L:
    for s,r in model.Upsilon_l[l]:
        for k in model.K_l[l]:
            for t in model.T:
                for b in model.B:
                    model.eq8.add(model.f_l_srktb[l,s,r,k,t,b] <= model.y_l_srkt[l,s,r,k,t]*model.Fup_l_k[l,k])
                    model.eq8.add(model.f_l_srktb[l,r,s,k,t,b] <= model.y_l_srkt[l,r,s,k,t]*model.Fup_l_k[l,k])
                        
model.eq9 = pyo.ConstraintList()
for tr in model.TR:
    for s in model.Omega_SS:
        for k in model.K_tr[tr]:
            for t in model.T:
                for b in model.B:
                    model.eq9.add(model.g_tr_sktb[tr,s,k,t,b] <= model.y_tr_skt[tr,s,k,t]*model.Gup_tr_k[tr,k])

model.eq10 = pyo.ConstraintList()
for t in model.T:
    for s in model.Omega_LN_t[t]:
        for b in model.B:
            model.eq10.add(model.d_U_stb[s,t,b] <= model.Mi__b[b]*model.D__st[s,t])

model.eq11 = pyo.ConstraintList()
for s in model.Omega_p["C"]:
    for k in model.K_p["C"]:
        for t in model.T:
            for b in model.B:
                model.eq11.add(model.g_p_sktb["C",s,k,t,b] <= model.y_p_skt["C",s,k,t]*model.Gup_p_k["C",k])

model.eq12 = pyo.ConstraintList()
for s in model.Omega_p["W"]:
    for k in model.K_p["W"]:
        for t in model.T:
            for b in model.B:
                model.eq12.add(model.g_p_sktb["W",s,k,t,b] <= model.y_p_skt["W",s,k,t]*min(model.Gup_p_k["W",k],model.Gmax_W_sktb[s,k,t,b]))

model.eq13 = pyo.ConstraintList()
for t in model.T:
    for b in model.B:
        model.eq13.add(sum(sum(sum(model.g_p_sktb[p,s,k,t,b]
                for s in model.Omega_p[p])
            for k in model.K_p[p])
        for p in model.P) 
        <= model.Vare*sum(model.Mi__b[b]*model.D__st[s,t]
            for s in model.Omega_LN_t[t])       
        )
   
model.eq14 = pyo.ConstraintList()
for t in model.T:
    for b in model.B:
        for s in model.Omega_N:
            model.eq14.add(sum(sum(sum(model.f_l_srktb[l,s,r,k,t,b] - model.f_l_srktb[l,r,s,k,t,b]
                        for r in model.Omega_l_s[l,s]) 
                    for k in model.K_l[l])
                for l in model.L) == - model.Mi__b[b]*model.D__st[s,t] + model.d_U_stb[s,t,b]
                        )

model.eq14_aux1 = pyo.ConstraintList()
for t in model.T:
    for b in model.B:
        for s in model.Omega_SS:
            model.eq14_aux1.add(sum(sum(sum(model.f_l_srktb[l,s,r,k,t,b] - model.f_l_srktb[l,r,s,k,t,b]
                        for r in model.Omega_l_s[l,s]) 
                    for k in model.K_l[l])
                for l in model.L) == (sum(sum(model.g_tr_sktb[tr,s,k,t,b]
                                for k in model.K_tr[tr])
                            for tr in model.TR)
                        )                            
                  )

model.eq14_aux2 = pyo.ConstraintList()
for t in model.T:
    for b in model.B:
        for s in model.Omega_p['C']:
            model.eq14_aux2.add(sum(sum(sum(model.f_l_srktb[l,s,r,k,t,b] - model.f_l_srktb[l,r,s,k,t,b]
                        for r in model.Omega_l_s[l,s]) 
                    for k in model.K_l[l])
                for l in model.L) == sum(model.g_p_sktb['C',s,k,t,b]
                            for k in model.K_p['C'])
                        )


model.eq14_aux3 = pyo.ConstraintList()
for t in model.T:
    for b in model.B:
        for s in model.Omega_p['W']:
            model.eq14_aux2.add(sum(sum(sum(model.f_l_srktb[l,s,r,k,t,b] - model.f_l_srktb[l,r,s,k,t,b]
                        for r in model.Omega_l_s[l,s]) 
                    for k in model.K_l[l])
                for l in model.L) == sum(model.g_p_sktb['W',s,k,t,b]
                            for k in model.K_p['W'])
                        )

# =============================================================================
# model.eq14_aux3 = pyo.ConstraintList() # It avoids "ET" transf. on new substations
# for t in T:
#     for b in B:
#         for s in Omega_SSN:
#             for k in K_tr['ET']:
#                 model.eq14_aux3.add(model.y_tr_skt['ET',s,k,t] == 0)
# 
# model.eq14_aux4 = pyo.ConstraintList() # It allows one type of transf. on existing substation nodes
# for t in T:
#     for s in Omega_SSE:
#         model.eq14_aux4.add(sum(sum(model.y_tr_skt[tr,s,k,t]
#                     for k in K_tr[tr])
#                 for tr in TR) <= 1
#             )
# =============================================================================


model.eq16_1 = pyo.ConstraintList()
for t in model.T:
    for b in model.B:
        for l in model.L:
            for s,r in model.Upsilon_l[l]:
                for k in model.K_l[l]:
                    model.eq16_1.add((-model.Z_l_k[l,k]*model.l__sr[s,r]*model.f_l_srktb[l,r,s,k,t,b]/model.Vbase + (model.V_stb[r,t,b] - model.V_stb[s,t,b]))
                                     <= model.H*(1-model.y_l_srkt[l,r,s,k,t]))
                    model.eq16_1.add((-model.Z_l_k[l,k]*model.l__sr[s,r]*model.f_l_srktb[l,s,r,k,t,b]/model.Vbase + (model.V_stb[s,t,b] - model.V_stb[r,t,b]))
                                     <= model.H*(1-model.y_l_srkt[l,s,r,k,t]))
                    
model.eq16_2 = pyo.ConstraintList()
for t in model.T:
    for b in model.B:
        for l in model.L:
            for s,r in model.Upsilon_l[l]:
                for k in model.K_l[l]:
                    model.eq16_2.add((model.Z_l_k[l,k]*model.l__sr[s,r]*model.f_l_srktb[l,r,s,k,t,b]/model.Vbase - (model.V_stb[r,t,b] - model.V_stb[s,t,b]))
                                     <= model.H*(1-model.y_l_srkt[l,r,s,k,t]))
                    model.eq16_2.add((model.Z_l_k[l,k]*model.l__sr[s,r]*model.f_l_srktb[l,s,r,k,t,b]/model.Vbase - (model.V_stb[s,t,b] - model.V_stb[r,t,b]))
                                     <= model.H*(1-model.y_l_srkt[l,s,r,k,t]))

# =============================================================================
# Investiment Constraints
# =============================================================================

model.eq17 = pyo.ConstraintList()
for l in model.L_nl:
    for s,r in model.Upsilon_l[l]:
        model.eq17.add(sum(sum(model.x_l_srkt[l,s,r,k,t]
                        for k in model.K_l[l])
                    for t in model.T) <= 1
        )
        
model.eq18 = pyo.ConstraintList()
for s in model.Omega_SS:
    model.eq18.add(sum(model.x_SS_st[s,t]
                       for t in model.T) <= 1
    )       
        
model.eq19 = pyo.ConstraintList()
for s in model.Omega_SS:
    model.eq19.add(sum(sum(model.x_NT_skt[s,k,t]
                for k in model.K_tr["NT"])
            for t in model.T) <= 1
        )        
        
model.eq20 = pyo.ConstraintList()
for p in model.P:
    for s in model.Omega_p[p]:
        model.eq20.add(sum(sum(model.x_p_skt[p,s,k,t]
                for k in model.K_p[p])
            for t in model.T) <= 1
        )        
        
model.eq21 = pyo.ConstraintList()
for s in model.Omega_SS:
    for k in model.K_tr["NT"]:
        for t in model.T:
            model.eq21.add(model.x_NT_skt[s,k,t] 
                           <=
                           sum(model.x_SS_st[s,y] 
                               for y in range(1, t+1))       
            )        
        
        
        
        
        
        