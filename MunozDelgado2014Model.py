"""
Created on Fri Nov 13 14:57:26 2020

Implementation on Pyomo of Distribution Expansion Planning Model proposed by Muñoz-Delgado et al. (2014).

Reference:
Muñoz-Delgado, G., Contreras, J., & Arroyo, J. M. (2014). Joint expansion planning of distributed generation and distribution networks. IEEE Transactions on Power Systems, 30(5), 2579-2590.
DOI: 10.1109/TPWRS.2014.2364960

@Code Athor: Jonas Villela de Souza
"""
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from Data_24Bus import *  
#from Data_138Bus import *    

# =============================================================================
# DG Penetration
# =============================================================================

Vare = 0.25 #Penetration limit for distributed generation.

# =============================================================================
# Model
# =============================================================================

model = pyo.ConcreteModel()

# =============================================================================
# Variables
# =============================================================================

model.C_E_t = pyo.Var(T, 
                      bounds=(0.0,None)
                      )

model.C_M_t = pyo.Var(T, 
                      bounds=(0.0,None)
                      )

model.C_R_t = pyo.Var(T, 
                      bounds=(0.0,None)
                      )

model.C_U_t = pyo.Var(T, 
                      bounds=(0.0,None)
                      )

model.C_I_t = pyo.Var(T, #Investment
                      bounds=(0.0,None)
                      )

model.C_TPV = pyo.Var(bounds=(0.0,None)
                      )

model.d_U_stb = pyo.Var(Omega_N, 
                        T,
                        B,
                        bounds=(0.0,None)
                        )

def f_l_rule(m):
    index = []
    for l in L:
        for s in Omega_N:
            for O in Omega_l_s[l][s-1]:
                for K in K_l[l]:
                    for t in T:
                        for b in B:
                            index.append((l,s,O,K,t,b))
    return index
            
model.f_l_rule = pyo.Set(dimen=6, initialize=f_l_rule)
model.f_l_srktb = pyo.Var(model.f_l_rule,
                          bounds=(0.0,None)
    )
model.ftio_l_srktb = pyo.Var(model.f_l_rule,
                          bounds=(0.0,None)
    )

def g_p_rule(m):
    index = []
    for p in P:
        for O in Omega_N:
            for kp in K_p[p]:
                for t in T:
                    for b in B:
                        index.append((p,O,kp,t,b))
    return index

model.g_p_rule = pyo.Set(dimen=5, initialize=g_p_rule)
model.g_p_sktb = pyo.Var(model.g_p_rule,
                         bounds=(0.0,None)                    
    )

def g_tr_rule(m):
    index = []
    for tr in TR:
        for O in Omega_N:
            for ktr in K_tr[tr]:
                for t in T:
                    for b in B:
                        index.append((tr,O,ktr,t,b))
    return index

model.g_tr_rule = pyo.Set(dimen=5, initialize=g_tr_rule)
model.g_tr_sktb = pyo.Var(model.g_tr_rule,
                         bounds=(0.0,None)                        
    )

model.gtio_SS_stb = pyo.Var(Omega_N, 
                            T, 
                            B,
                            bounds=(0.0,None)
    )

model.V_stb = pyo.Var(Omega_N, 
                            T, 
                            B,
                            bounds=(0.0,None)
    )

def x_l_rule(m):
    index = []
    for l in ["NRF", "NAF"]:
        for s in Omega_N:
            for O in Omega_l_s[l][s-1]:
                for K in K_l[l]:
                    for t in T:
                        index.append((l,s,O,K,t))
    return index

model.x_l_rule = pyo.Set(dimen=5, initialize=x_l_rule)
model.x_l_srkt = pyo.Var(model.x_l_rule,
                         within=pyo.Binary
    )
                
def x_NT_rule(m):
    index = []
    for SS in Omega_SS:
        for k in K_tr["NT"]:
            for t in T:
                index.append((SS,k,t))
    return index

model.x_NT_rule = pyo.Set(dimen=3, initialize=x_NT_rule)
model.x_NT_skt = pyo.Var(model.x_NT_rule, 
                         within=pyo.Binary
    )

def x_p_rule(m):
    index = []
    for p in P:
        for O in Omega_p[p]:
            for K in K_p[p]:
                for t in T:
                    index.append((p,O,K,t))
    return index 

model.x_p_rule = pyo.Set(dimen=4, initialize=x_p_rule)
model.x_p_skt = pyo.Var(model.x_p_rule,
                        within=pyo.Binary
    )

def x_SS_rule(model):
    index = []
    for SS in Omega_SS:
        for t in T:
            index.append((SS,t))
    return index
model.x_SS_rule = pyo.Set(dimen=2, initialize=x_SS_rule)
model.x_SS_st = pyo.Var(model.x_SS_rule,
                        within=pyo.Binary
                        )


def y_l_rule(m):
    index = []
    for l in L:
        for s in Omega_N:
            for O in Omega_l_s[l][s-1]:
                for K in K_l[l]:
                    for t in T:
                        index.append((l,s,O,K,t))
    return index 

model.y_l_rule = pyo.Set(dimen=5, initialize=y_l_rule)
model.y_l_srkt = pyo.Var(model.y_l_rule,
                         within=pyo.Binary
    )

def y_p_rule(m):
    index = []
    for p in P:
        for O in Omega_N:
            for K in K_p[p]:
                for t in T:
                    index.append((p,O,K,t))
    return index 

model.y_p_rule = pyo.Set(dimen=4, initialize=y_p_rule)
model.y_p_skt = pyo.Var(model.y_p_rule,
                        within=pyo.Binary
    )

def y_tr_rule(m):
    index = []
    for tr in TR:
        for O in Omega_N:
            for K in K_tr[tr]:
                for t in T:
                    index.append((tr,O,K,t))
    return index 

model.y_tr_rule = pyo.Set(dimen=4, initialize=y_tr_rule)
model.y_tr_skt = pyo.Var(model.y_tr_rule,
                         within=pyo.Binary
    )

def delta_l_rule(m):
    index = []
    for l in L:
        for s in Omega_N:
            for O in Omega_l_s[l][s-1]:
                for K in K_l[l]:
                    for t in T:
                        for b in B:
                            for V in range(1,n__V+1):
                                index.append((l,s,O,K,t,b,V))
    return index 

model.delta_l_rule = pyo.Set(dimen=7, initialize=delta_l_rule)
model.delta_l_srktbv = pyo.Var(model.delta_l_rule,
                               bounds=(0.0,None)                             
    )

def delta_tr_rule(m):
    index = []
    for tr in TR:
        for O in Omega_SS:
            for K in K_tr[tr]:
                for t in T:
                    for b in B:
                        for V in range(1,n__V+1):
                            index.append((tr,O,K,t,b,V))
    return index 

model.delta_tr_rule = pyo.Set(dimen=6, initialize=delta_tr_rule)
model.delta_tr_sktbv = pyo.Var(model.delta_tr_rule,
                               bounds=(0.0,None)
    )


# =============================================================================
# Objective Function
# =============================================================================

model.Obj = pyo.Objective(expr=model.C_TPV, sense=pyo.minimize)


# =============================================================================
# Costs Constraints
# =============================================================================

def C_TPV_rule(m):
    return model.C_TPV == (sum(model.C_I_t[t]*(((1+i)**(-t))/i) 
                               for t in T)
                           + sum((model.C_M_t[t] + model.C_E_t[t] + model.C_R_t[t] + model.C_U_t[t])*((1+i)**(-t)) 
                               for t in T)
                           + ((model.C_M_t[T[-1]] + model.C_E_t[T[-1]] + model.C_R_t[T[-1]] + model.C_U_t[T[-1]])*((1+i)**(-T[-1])/i))
                           )
model.eq1 = pyo.Constraint(rule=C_TPV_rule)

def eq2_rule(model,t):
    return model.C_I_t[t] == (sum(RR_l[l]*sum(sum(C_Il_k[l][k-1]*l__sr[s-1,r-1]*model.x_l_srkt[l,s,r,k,t] 
                                            for s,r in Upsilon_l[l])
                                        for k in K_l[l])
                            for l in ["NRF", "NAF"])
                            
                            + RR_SS*sum(C_ISS_s[s]*model.x_SS_st[s,t] 
                                        for s in Omega_SS) 
                              
                            + RR_NT*sum(sum(C_INT_k[k-1]*model.x_NT_skt[s,k,t]
                                    for s in Omega_SS)
                                for k in K_tr["NT"]) 
                            
                            + sum(RR_p[p]*sum(sum(C_Ip_k[p][k-1]*pf*Gup_p_k[p][k-1]*model.x_p_skt[p,s,k,t]
                                        for s in Omega_p[p])
                                    for k in K_p[p])
                                for p in P)                              
                            )    
model.eq2 = pyo.Constraint(T, rule=eq2_rule)

def eq3_rule(model,t):
    return model.C_M_t[t] == (sum(sum(sum(C_Ml_k[l][k-1]*(model.y_l_srkt[l,s,r,k,t] - model.y_l_srkt[l,r,s,k,t])
                    for s,r in Upsilon_l[l])
                for k in K_l[l])
            for l in L)
            
            + sum(sum(sum(C_Mtr_k[tr][k-1]*model.y_tr_skt[tr,s,k,t]
                    for s in Omega_SS)
                for k in K_tr[tr])
            for tr in TR)
            
            + sum(sum(sum(C_Mp_k[p][k-1]*model.y_p_skt[p,s,k,t]
                    for s in Omega_p[p])
                for k in K_p[p])
            for p in P)
        
        )
model.eq3 = pyo.Constraint(T, rule=eq3_rule)

def eq4_rule(model,t):
    return model.C_E_t[t] == (sum(Delta__b[b-1]*pf*(sum(sum(sum(C_SS_b[b-1]*model.g_tr_sktb[tr,s,k,t,b]
                                for s in Omega_SS)
                            for k in K_tr[tr])
                        for tr in TR)
                        
                        + sum(sum(sum(C_Ep_k[p][k-1]*model.g_p_sktb[p,s,k,t,b]
                                for s in Omega_p[p])
                            for k in K_p[p])
                        for p in P)
                        )
                for b in B)
            )
model.eq4 = pyo.Constraint(T, rule=eq4_rule)

def eq5_rule(model,t):
    return model.C_R_t[t] == (sum(Delta__b[b-1]*C_SS_b[b-1]*pf*(sum(sum(sum(sum(
                                M_tr_kV[tr][k-1][y-1]*model.delta_tr_sktbv[tr,s,k,t,b,y]
                                for y in range(1,n__V+1))
                            for s in Omega_SS)
                        for k in K_tr[tr])
                    for tr in TR)

                    + sum(sum(sum(sum(M_l_kV[l][k-1][z-1]*l__sr[s-1,r-1]*(model.delta_l_srktbv[l,s,r,k,t,b,z] + model.delta_l_srktbv[l,r,s,k,t,b,z])
                                for z in range(1,n__V+1))
                            for s, r in Upsilon_l[l])
                        for k in K_l[l])
                    for l in L)
                )
            for b in B)
        )
model.eq5 = pyo.Constraint(T, rule=eq5_rule)

model.eq5_aux1 = pyo.ConstraintList()
for tr in TR:
    for s in Omega_SS:
        for k in K_tr[tr]:
            for t in T:
                for b in B:
                    model.eq5_aux1.add(model.g_tr_sktb[tr,s,k,t,b] == sum(model.delta_tr_sktbv[tr,s,k,t,b,V] for V in range(1,n__V+1)))

model.eq5_aux2 = pyo.ConstraintList()
for tr in TR:
    for s in Omega_SS:
        for k in K_tr[tr]:
            for t in T:
                for b in B:
                    for V in range(1,n__V+1):
                        model.eq5_aux2.add(model.delta_tr_sktbv[tr,s,k,t,b,V] <= A_tr_kV[tr][k-1][V-1])

model.eq5_aux3 = pyo.ConstraintList()
for l in L:
    for r in Omega_N:
        for s in Omega_l_s[l][r-1]:
            for k in K_l[l]:
                for t in T:
                    for b in B:
                        model.eq5_aux3.add(model.f_l_srktb[l,s,r,k,t,b] == sum(model.delta_l_srktbv[l,s,r,k,t,b,v] for v in range(1,n__V+1)))

model.eq5_aux4 = pyo.ConstraintList()
for l in L:
    for r in Omega_N:
        for s in Omega_l_s[l][r-1]:
            for k in K_l[l]:
                for t in T:
                    for b in B:
                        for v in range(1,n__V+1):
                            model.eq5_aux4.add(model.delta_l_srktbv[l,s,r,k,t,b,v] <= A_l_kV[l][k-1][v-1])

def eq6_rule(model,t):
    return model.C_U_t[t] == (sum(sum(Delta__b[b-1]*C_U*pf*model.d_U_stb[s,t,b]
                for s in Omega_LN_t[t])
            for b in B)
        )
model.eq6 = pyo.Constraint(T, rule=eq6_rule)

# =============================================================================
# Kirchhoff's Laws and Operational Limits
# =============================================================================

def eq7_rule(model, s, t, b):
    return pyo.inequality(V_ ,model.V_stb[s,t,b], Vup)
model.eq7 = pyo.Constraint(Omega_N, T, B, rule=eq7_rule)

model.eq8 = pyo.ConstraintList()
for l in L:
    for r in Omega_N:
        for s in Omega_l_s[l][r-1]:
            for k in K_l[l]:
                for t in T:
                    for b in B:
                        model.eq8.add(model.f_l_srktb[l,s,r,k,t,b] <= model.y_l_srkt[l,s,r,k,t]*Fup_l_k[l][k-1])
                        
model.eq9 = pyo.ConstraintList()
for tr in TR:
    for s in Omega_N:
        for k in K_tr[tr]:
            for t in T:
                for b in B:
                    model.eq9.add(model.g_tr_sktb[tr,s,k,t,b] <= model.y_tr_skt[tr,s,k,t]*Gup_tr_k[tr][k-1])

model.eq10 = pyo.ConstraintList()
for t in T:
    for s in Omega_N:
        for b in B:
            model.eq10.add(model.d_U_stb[s,t,b] <= Mi__b[b-1]*D__st[s-1,t-1])

model.eq11 = pyo.ConstraintList()
for s in Omega_N:
    for k in K_p["C"]:
        for t in T:
            for b in B:
                model.eq11.add(model.g_p_sktb["C",s,k,t,b] <= model.y_p_skt["C",s,k,t]*Gup_p_k["C"][k-1])

model.eq12 = pyo.ConstraintList()
for s in Omega_N:
    for k in K_p["W"]:
        for t in T:
            for b in B:
                model.eq12.add(model.g_p_sktb["W",s,k,t,b] <= model.y_p_skt["W",s,k,t]*min(Gup_p_k["W"][k-1],Gmax_W_sktb[s-1,k-1,t-1,b-1]))

model.eq13 = pyo.ConstraintList()
for t in T:
    for b in B:
        model.eq13.add(sum(sum(sum(model.g_p_sktb[p,s,k,t,b]
                for s in Omega_p[p])
            for k in K_p[p])
        for p in P) 
        <= Vare*sum(Mi__b[b-1]*D__st[s-1,t-1]
            for s in Omega_LN_t[t])       
        )
   
model.eq14 = pyo.ConstraintList()
for t in T:
    for b in B:
        for s in Omega_N:
            model.eq14.add(sum(sum(sum(model.f_l_srktb[l,s,r,k,t,b] - model.f_l_srktb[l,r,s,k,t,b]
                        for r in Omega_l_s[l][s-1]) 
                    for k in K_l[l])
                for l in L) == (sum(sum(model.g_tr_sktb[tr,s,k,t,b]
                                for k in K_tr[tr])
                            for tr in TR)
                            + sum(sum(model.g_p_sktb[p,s,k,t,b]
                                for k in K_p[p])
                            for p in P)
                            - Mi__b[b-1]*D__st[s-1,t-1]
                            + model.d_U_stb[s,t,b]
                        )
                )
                                        
model.eq14_aux1 = pyo.ConstraintList() #It allows DG only on candidates nodes
for t in T:
    for p in P:
        for k in K_p[p]:
            for s in Omega_N:
                if s not in Omega_p[p]:
                    model.eq14_aux1.add(model.y_p_skt[p,s,k,t] == 0)
                    
model.eq14_aux2 = pyo.ConstraintList() #It allows transf. only on candidates nodes
for t in T:
    for tr in TR:
        for k in K_tr[tr]:
            for s in Omega_N:
                if s not in Omega_SS:
                    model.eq14_aux2.add(model.y_tr_skt[tr,s,k,t] == 0)

model.eq14_aux3 = pyo.ConstraintList() # It avoids "ET" transf. on new substations
for t in T:
    for b in B:
        for s in Omega_SSN:
            for k in K_tr['ET']:
                model.eq14_aux3.add(model.y_tr_skt['ET',s,k,t] == 0)

model.eq14_aux4 = pyo.ConstraintList() # It allows one type of transf. on existing substation nodes
for t in T:
    for s in Omega_SSE:
        model.eq14_aux4.add(sum(sum(model.y_tr_skt[tr,s,k,t]
                    for k in K_tr[tr])
                for tr in TR) <= 1
            )

model.eq16_1 = pyo.ConstraintList()
for t in T:
    for b in B:
        for l in L:
            for r in Omega_N:
                for s in Omega_l_s[l][r-1]:
                    for k in K_l[l]:
                        model.eq16_1.add((-Z_l_k[l][k-1]*l__sr[s-1,r-1]*model.f_l_srktb[l,s,r,k,t,b]/Vbase + (model.V_stb[s,t,b] - model.V_stb[r,t,b]))
                                         <= H*(1-model.y_l_srkt[l,s,r,k,t]))

model.eq16_2 = pyo.ConstraintList()
for t in T:
    for b in B:
        for l in L:
            for r in Omega_N:
                for s in Omega_l_s[l][r-1]:
                    for k in K_l[l]:
                        model.eq16_2.add((Z_l_k[l][k-1]*l__sr[s-1,r-1]*model.f_l_srktb[l,s,r,k,t,b]/Vbase - (model.V_stb[s,t,b] - model.V_stb[r,t,b]))
                                         <= H*(1-model.y_l_srkt[l,s,r,k,t]))
                            
# =============================================================================
# Investiment Constraints
# =============================================================================

model.eq17 = pyo.ConstraintList()
for l in ["NRF", "NAF"]:
    for s,r in Upsilon_l[l]:
        model.eq17.add(sum(sum(model.x_l_srkt[l,s,r,k,t]
                        for k in K_l[l])
                    for t in T) <= 1
        )
        
model.eq18 = pyo.ConstraintList()
for s in Omega_SS:
    model.eq18.add(sum(model.x_SS_st[s,t]
                       for t in T) <= 1
    )

model.eq19 = pyo.ConstraintList()
for s in Omega_SS:
    model.eq19.add(sum(sum(model.x_NT_skt[s,k,t]
                for k in K_tr["NT"])
            for t in T) <= 1
        )

model.eq20 = pyo.ConstraintList()
for p in P:
    for s in Omega_p[p]:
        model.eq20.add(sum(sum(model.x_p_skt[p,s,k,t]
                for k in K_p[p])
            for t in T) <= 1
        )

model.eq21 = pyo.ConstraintList()
for s in Omega_SS:
    for k in K_tr["NT"]:
        for t in T:
            model.eq21.add(model.x_NT_skt[s,k,t] 
                           <=
                           sum(model.x_SS_st[s,y] 
                               for y in range(1,t+1))       
            )

#Eq. updated #Ref: DOI: 10.1109/TSG.2016.2560339
model.eq22 = pyo.ConstraintList()
for t in T:
    for l in ["EFF"]:
        for k in K_l[l]:
            for s,r in Upsilon_l[l]:
                model.eq22.add(model.y_l_srkt[l,s,r,k,t] + model.y_l_srkt[l,r,s,k,t]
                               == 1
                )

#Eq. updated #Ref: DOI: 10.1109/TSG.2016.2560339
model.eq23 = pyo.ConstraintList()
for t in T:
    for l in ["NRF", "NAF"]:
        for k in K_l[l]:
            for s,r in Upsilon_l[l]:
                model.eq23.add(model.y_l_srkt[l,s,r,k,t] + model.y_l_srkt[l,r,s,k,t] 
                               == sum(model.x_l_srkt[l,s,r,k,y]
                                   for y in range(1,t+1))
                )

#Eq. updated #Ref: DOI: 10.1109/TSG.2016.2560339
model.eq24 = pyo.ConstraintList()
for t in T:
    for l in ["ERF"]:
        for k in K_l[l]:
            for s,r in Upsilon_l[l]:
                model.eq24.add(model.y_l_srkt[l,s,r,k,t] + model.y_l_srkt[l,r,s,k,t] 
                               == 1 - sum(sum(model.x_l_srkt["NRF",s,r,z,y]
                                       for z in K_l["NRF"])
                                   for y in range(1,t+1))
                )

model.eq25 = pyo.ConstraintList()
for t in T:
    for s in Omega_SS:
        for k in K_tr["NT"]:
            model.eq25.add(model.y_tr_skt["NT", s, k, t] 
                           <= sum(model.x_NT_skt[s,k,y]
                               for y in range(1,t+1))
                           )

model.eq26 = pyo.ConstraintList()
for t in T:
    for p in P:
        for s in Omega_p[p]:
            for k in K_p[p]:
                model.eq26.add(model.y_p_skt[p,s,k,t] <=
                               sum(model.x_p_skt[p,s,k,y]
                                   for y in range(1,t+1))
                               )

def eq27_rule(model,t):
    return ((sum(sum(sum(C_Il_k[l][k-1]*l__sr[s-1,r-1]*model.x_l_srkt[l,s,r,k,t]
                for s,r in Upsilon_l[l])
            for k in K_l[l])
        for l in ["NRF", "NAF"])
        + sum(C_ISS_s[s]*model.x_SS_st[s,t]
            for s in Omega_SS)
        + sum(sum(C_INT_k[k-1]*model.x_SS_st[s,t]
                for s in Omega_SS)
            for k in K_tr["NT"])
        + sum(sum(sum(C_Ip_k[p][k-1]*pf*Gup_p_k[p][k-1]*model.x_p_skt[p,s,k,t]
                    for s in Omega_p[p])
                for k in K_p[p])
            for p in P)
        <= IB__t[t-1])
    )

model.eq27 = pyo.Constraint(T, rule=eq27_rule)

# =============================================================================
# Radiality Constraints
# =============================================================================
model.eq28 = pyo.ConstraintList()
for t in T:
    for r in Omega_LN_t[t]:
        model.eq28.add(sum(sum(sum(model.y_l_srkt[l,s,r,k,t] 
                    for k in K_l[l])
                for s in Omega_l_s[l][r-1])
            for l in L) == 1
        )

model.eq29 = pyo.ConstraintList()
for t in T:
    for r in Omega_N:
        if r not in Omega_LN_t[t]:
            model.eq29.add(sum(sum(sum(model.y_l_srkt[l,s,r,k,t] 
                        for k in K_l[l])
                    for s in Omega_l_s[l][r-1])
                for l in L) <= 1
            )

model.eq30 = pyo.ConstraintList()
for t in T:
    for b in B:
        for s in Omega_N:
            model.eq30.add(sum(sum(sum(model.ftio_l_srktb[l,s,r,k,t,b] - model.ftio_l_srktb[l,r,s,k,t,b]
                        for r in Omega_l_s[l][s-1])
                    for k in K_l[l])
                for l in L) == model.gtio_SS_stb[s,t,b] - Dtio_stb[s-1,t-1,b-1]
            )
                           
model.eq31 = pyo.ConstraintList()                
for t in T:
    for b in B:
        for l in ["EFF"]:
            for r in Omega_N:
                for s in Omega_l_s[l][r-1]:
                    for k in K_l[l]:
                        model.eq31.add(model.ftio_l_srktb[l,s,r,k,t,b] <= n__DG)
                    
model.eq32 = pyo.ConstraintList()
for t in T:
    for b in B:
        for l in ["ERF"]:
            for s,r in Upsilon_l[l]:
                for k in K_l[l]:
                    model.eq32.add(model.ftio_l_srktb[l,s,r,k,t,b] <= n__DG*(
                        1 - sum(sum(model.x_l_srkt["NRF",s,r,z,y]
                                for z in K_l["NRF"])
                             for y in range(1,t+1))
                        )
                    )

model.eq33 = pyo.ConstraintList()
for t in T:
    for b in B:
        for l in ["ERF"]:
            for s,r in Upsilon_l[l]:
                for k in K_l[l]:
                    model.eq33.add(model.ftio_l_srktb[l,r,s,k,t,b] <= n__DG*(
                        1 - sum(sum(model.x_l_srkt["NRF",s,r,z,y]
                                for z in K_l["NRF"])
                             for y in range(1,t+1))
                        )
                    )

model.eq34 = pyo.ConstraintList()
for t in T:
    for b in B:
        for l in ["NRF", "NAF"]:
            for k in K_l[l]:
                for s,r in Upsilon_l[l]:
                    model.eq34.add(model.ftio_l_srktb[l,s,r,k,t,b] <= n__DG*(
                        sum(model.x_l_srkt[l,s,r,k,y]
                            for y in range(1,t+1))
                        )
                    )

model.eq35 = pyo.ConstraintList()
for t in T:
    for b in B:
        for l in ["NRF", "NAF"]:
            for k in K_l[l]:
                for s,r in Upsilon_l[l]:
                    model.eq35.add(model.ftio_l_srktb[l,r,s,k,t,b] <= n__DG*(
                        sum(model.x_l_srkt[l,s,r,k,y]
                            for y in range(1,t+1))
                        )
                    )

model.eq36 = pyo.ConstraintList()
for t in T:
    for b in B:
        for s in Omega_SS:
            model.eq36.add(model.gtio_SS_stb[s,t,b] <= n__DG)

model.eq36_aux = pyo.ConstraintList()
for t in T:
    for b in B:
        for s in Omega_N:
            if s not in Omega_SS:
                model.eq36_aux.add(model.gtio_SS_stb[s,t,b] == 0)

# =============================================================================
# Solver
# =============================================================================

opt = SolverFactory('cplex')
opt.options['threads'] = 16
opt.options['mipgap'] = 1/100
opt.solve(model, warmstart=False, tee=True)

# =============================================================================
# Results: Reports
# =============================================================================

#Results - 
Yearly_Costs = []
for i in range(1,np.shape(T)[0]+1):
    year_aux = {
                'Investment':np.round(pyo.value(model.C_I_t[i])/1e6,4),
                'Maintenance':np.round(pyo.value(model.C_M_t[i])/1e6,4),
                'Production':np.round(pyo.value(model.C_E_t[i])/1e6,4),
                'Losses':np.round(pyo.value(model.C_R_t[i])/1e6,4),
                'Unserved_energy':np.round(pyo.value(model.C_U_t[i])/1e6,4)
        }
    Yearly_Costs.append(year_aux)
Yearly_Costs = pd.DataFrame(Yearly_Costs)

#Binary utilization variables for feeders
Variable_Util_l = []
for l in L: #Type of line
    for s in Omega_N: #Buses from 
        for r in Omega_l_s[l][s-1]: #Buses to
            for k in K_l[l]: #Line option 
                for t in T: #Time stage
                    var_aux ={
                        'T_Line': l,
                        'From': s,
                        'To': r,
                        'Option': k,
                        'Stage': t,
                        'Decision': pyo.value(model.y_l_srkt[l,s,r,k,t])
                        }
                    Variable_Util_l.append(var_aux)
Variable_Util_l = pd.DataFrame(Variable_Util_l)

#Binary utilization variables for transformers
Variable_Util_tr = []
for tr in TR:
    for s in Omega_N:
        for k in K_tr[tr]:
            for t in T:
                var_aux ={
                    "Trans_T":tr,
                    "Bus":s,
                    "Option":k,
                    "Stage":t,
                    "Decision":pyo.value(model.y_tr_skt[tr,s,k,t])
                    }
                Variable_Util_tr.append(var_aux)
Variable_Util_tr = pd.DataFrame(Variable_Util_tr)

#Current injections corresponding to transformers
Current_inj_TR = []
for tr in TR:
    for s in Omega_N:
        for k in K_tr[tr]:
            for t in T:
                for b in B:
                    aux = {
                        "TR_Type": tr,
                        "Bus": s,
                        "Option": k,
                        "Stage": t,
                        "Load_l": b,
                        "Injection": pyo.value(model.g_tr_sktb[tr,s,k,t,b] )
                        }
                    Current_inj_TR.append(aux)
Current_inj_TR = pd.DataFrame(Current_inj_TR)

#Actual current flows through feeders
Actual_C_Flow_l = []
for l in L: #Type of line
    for s in Omega_N: #Buses from 
        for r in Omega_l_s[l][s-1]: #Buses to
            for k in K_l[l]: #Line option 
                for t in T: #Time stage 
                    for b in B: #Load level
                        if pyo.value(model.f_l_srktb[l,s,r,k,t,b]) > 0.1:
                            actual_aux = {
                                'T_Line': l,
                                'From': s,
                                'To': r,
                                'Option': k,
                                'Stage': t,
                                'L_level': b,
                                'Flow': pyo.value(model.f_l_srktb[l,s,r,k,t,b])
                                }
                            Actual_C_Flow_l.append(actual_aux) 
Actual_C_Flow_l = pd.DataFrame(Actual_C_Flow_l)                            
