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

# =============================================================================
# def power_out(k,speed):
#     if k == 1:
#         coef = (0.91-0.02)/(15-4)
#         if speed < 4:
#             return 0
#         elif speed <= 15 and speed >= 4:
#             return speed*coef
#         else:
#             return 0.91
#     elif k == 2:
#         coef = (2.05-0.025)/(12-3)
#         if speed < 3:
#             return 0
#         elif speed <= 12 and speed >= 3:
#             return speed*coef
#         else:
#             return 2.05
# =============================================================================
    
def power_out(k,speed):
    if k == 1:
        WG = np.array([[3, 4.0],
                       [4, 20.0],
                       [5, 50.0],
                       [6, 96.0],
                       [7, 156.0],
                       [8, 238.0],
                       [9, 340.0],
                       [10, 466.0],
                       [11, 600.0],
                       [12, 710.0],
                       [13, 790.0],
                       [14, 850.0],
                       [15, 880.0],
                       [16, 905.0],
                       [17, 910.0]]
                      )
    elif k == 2:
        WG = np.array([[2, 3.0],
                       [3, 25.0], 
                       [4, 82.0],
                       [5, 174.0],
                       [6, 321.0],
                       [7, 532.0], 
                       [8, 815.0],
                       [9, 1180.0],
                       [10, 1580.0],
                       [11, 1810.0],
                       [12, 1980.0],
                       [13, 2050.0]]
                      )
        
    if k == 1 and speed < 3:
        Pr = 0
    elif k == 1 and speed >= 17:
        Pr = 0.91 
    elif k == 2 and speed < 2:
        Pr = 0
    elif k == 2 and speed >= 13:
        Pr = 2.05
    else:
        speed_aux1 = int(speed)
        speed_aux2 = speed_aux1 + 1
        
        loc_aux1 = np.where(speed_aux1 == WG[:,0])[0].item()
        loc_aux2 = np.where(speed_aux2 == WG[:,0])[0].item()
        
        Pr_aux1 = (speed*WG[loc_aux1,1])/speed_aux1
        Pr_aux2 = (speed*WG[loc_aux2,1])/speed_aux2
        
        Pr = ((Pr_aux1+Pr_aux2)/2)/1000
       
    return Pr

# =============================================================================
# System Data
# =============================================================================

n_bus = 24  #Number of buses
n_branches = 33 #Number of branches

load_factor = [0.7, 0.83, 1]

#EFF = Existing Fixed Feeder
#ERF = Existing Replaceable Feeder
#NRF = New Replacement Feeder
#NAF = New Added Feeder

branch = [ #(s,r) length  type (km)
          ((1,5), 2.22, "NAF"), #Ok 
          ((1,9), 1.2, "NAF"), #Ok
          ((1,14), 1.2, "NAF"), #Ok
          ((1,21), 2.2, "ERF"), #Ok
          ((2,3), 2.0, "NAF"), #Ok
          ((2,12), 1.1, "NAF"), #Ok
          ((2,21), 1.7, "EFF"), #Ok 
          ((3,10), 1.1, "NAF"), #Ok
          ((3,16), 1.2, "NAF"), #Ok
          ((3,23), 1.2, "NAF"), #Ok
          ((4,7), 2.6, "NAF"), #Ok
          ((4,9), 1.2, "NAF"), #Ok
          ((4,15), 1.6, "NAF"), #Ok
          ((4,16), 1.3, "NAF"), #Ok
          ((5,6), 2.4, "NAF"), #Ok
          ((5,24), 0.7, "NAF"), #OK
          ((6,13), 1.2, "NAF"), #OK
          ((6,17), 2.2, "NAF"), #Ok
          ((6,22), 2.7, "EFF"), #Ok 
          ((7,8), 2.0, "NAF"), #Ok
          ((7,11), 1.1, "NAF"), #Ok
          ((7,19), 1.2, "NAF"), #Ok
          ((7,23), 0.9, "NAF"), #Ok
          ((8,22), 1.9, "ERF"), #Ok
          ((10,16), 1.6, "NAF"), #Ok
          ((10,23), 1.3, "NAF"), #Ok
          ((11,23), 1.6, "NAF"), #Ok
          ((14,18), 1.0, "NAF"), #Ok
          ((15,17), 1.2, "NAF"), #Ok
          ((15,19), 0.8, "NAF"), #Ok
          ((17,22), 1.5, "NAF"), #Ok
          ((18,24), 1.5, "NAF"), #Ok
          ((20,24), 0.9, "NAF") #Ok        
          ]


peak_demand = np.array([#Stages
                        #1     #2    #3
                        [4.05, 3.45, 5.42],
                        [0.78, 0.77, 1.21],
                        [2.58, 3.38, 3.98],
                        [0.32, 0.41, 2.43],
                        [0.28, 0.37, 0.47],
                        [1.17, 0.92, 1.81],
                        [4.04, 3.70, 4.36],
                        [0.72, 0.60, 0.94],
                        [1.14, 1.12, 1.77],
                        [1.56, 2.04, 2.40],
                        [0.00, 1.91, 2.80],
                        [0.00, 0.93, 1.29],
                        [0.00, 1.15, 1.87],
                        [0.00, 3.05, 3.16],
                        [0.00, 1.62, 1.62],
                        [0.00, 2.16, 1.22],
                        [0.00, 0.00, 2.40],
                        [0.00, 0.00, 2.10],
                        [0.00, 0.00, 1.81],
                        [0.00, 0.00, 3.79],
                        [0.00, 0.00, 0.00], #add eq14 problem
                        [0.00, 0.00, 0.00], #add eq14 problem
                        [0.00, 0.00, 0.00], #add eq14 problem
                        [0.00, 0.00, 0.00]  #add eq14 problem                       
                        ])

#Zones A = 1, B = 2, C = 3
               #Buses= 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
node_zone = np.array([[2, 3, 3, 1, 1, 2, 3, 3, 1, 2, 3, 3, 3, 2, 2, 2, 3, 1, 3, 1, 3, 3, 3, 1]])

wind_speed = np.array([#Load Level (m/s)
                       #1    2     3   
                      [8.53, 9.12, 10.04], #Zone A
                      [6.13, 7.26, 7.11],  #Zone B
                      [4.13, 5.10, 5.56]   #Zone C
                      ])

# =============================================================================
# Sets of Indexes
# =============================================================================

B = np.arange(1, len(load_factor)+1, dtype=int) #Set of Load Levels 
T = np.arange(1, np.shape(peak_demand)[1]+1, dtype=int) #Set of Time Stages
L = ["EFF", "ERF", "NRF", "NAF"]                        #Set of Feeder Types
#C = Conventional 
#W = Wind Generation 
P = ["C", "W"]                                          #Set of Generator Types
#ET = Existing Transformer
#NT = New Transformer
TR = ["ET", "NT"]                                       #Set of Transformer Types

# =============================================================================
# Sets of Alternatives
# =============================================================================

K_l = {"EFF": [1], #Sets of available alternatives for feeders
       "ERF": [1], 
       "NRF": [1, 2], 
       "NAF": [1, 2]
       } 

K_p = {"C": [1, 2], #Sets of available alternatives for generators
       "W": [1, 2]
       } 

K_tr = {"ET": [1], #Sets of available alternatives for transformers
       "NT": [1, 2]
       } 

# =============================================================================
# Sets of Branches
# =============================================================================

Upsilon_l = {"EFF": [],
             "ERF": [],
             "NRF": [],
             "NAF": []
             }

for branch_type in L: #Set of branches with feeders of type l
    for b in branch:
        if b[2] == branch_type:
            s = b[0][0]
            r = b[0][1]
            Upsilon_l[branch_type].append((s,r))
Upsilon_l["NRF"] = Upsilon_l["ERF"]

# =============================================================================
# Sets of Nodes
# =============================================================================

Omega_SS = [21, 22, 24, 23] #Sets of nodes connected to node s by substation nodes
Omega_SSE = [21, 22] # Fixing eq14
Omega_SSN = [23, 24] # Fixing eq14

Omega_l_s = {"EFF": [[] for i in range(0,n_bus)], #Sets of nodes connected to node s by a feeder of type l
             "ERF": [[] for i in range(0,n_bus)],
             "NRF": [[] for i in range(0,n_bus)],
             "NAF": [[] for i in range(0,n_bus)]
             }

for branch_type in L:
    for (s,r) in Upsilon_l[branch_type]:
        Omega_l_s[branch_type][(s,r)[0]-1].append((s,r)[1])
        Omega_l_s[branch_type][(s,r)[1]-1].append((s,r)[0])

Omega_LN_t = {1: [indx+1 for indx,value in enumerate(peak_demand[:, 0]) if value > 0], #Sets of nodes connected to node s by load nodes
              2: [indx+1 for indx,value in enumerate(peak_demand[:, 1]) if value > 0],
              3: [indx+1 for indx,value in enumerate(peak_demand[:, 2]) if value > 0]
              }

Omega_N = np.arange(1, n_bus+1, dtype=int) #Sets of nodes connected to node s by system nodes
Omega_p = {"C": [2, 3, 7, 13, 15, 16, 17, 20], #Sets of nodes connected to node s by distributed generation
           "W": [1, 4, 5, 9, 15, 17, 18, 19]
           }

# =============================================================================
# Energy Costs
# =============================================================================

#Load Levels
#         1     2   3
C_SS_b = [57.7, 70, 85.3] #the costs of the energy supplied by all substations

#DG units
C_Ep_k = {"C": [47, 45], #Conventional DG
          "W": [0, 0]    #Windy DG
          }

#Cost for unserved energy 
C_U = 2000

# =============================================================================
# Investment Costs
# =============================================================================

C_Il_k = {"NRF": [19140, 29870], #Investment cost coefficients of feeders
          "NAF": [15020, 25030]
          }

C_INT_k = [750000, 950000] #Investment cost coefficients of new transformers

C_Ip_k = {"C": [500000, 490000],  #Investment cost coefficients of generators
          "W": [1850000, 1840000]
          }

C_ISS_s = {21: 100000,  #Investment cost coefficients of substations
         22: 100000, 
         23: 140000, 
         24: 180000
         }

# =============================================================================
# Maintenance Costs
# =============================================================================

C_Ml_k = {"EFF": [450], #Maintenance cost coefficients of feeders
          "ERF": [450],
          "NRF": [450, 450],
          "NAF": [450, 450]
          }

C_Mp_k = {"C": [0.05*0.9*500000*1, 0.05*0.9*490000*2], #Maintenance cost coefficients of generators
          "W": [0.05*0.9*1850000*0.91, 0.05*0.9*1840000*2.05]
          }

C_Mtr_k = {"ET": [1000], #Maintenance cost coefficients of transformers
           "NT": [2000, 3000]
           }


# =============================================================================
# System's Data
# =============================================================================

D__st = peak_demand #Actual nodal peak demand

Dtio_stb = np.full((np.shape(Omega_N)[0],np.shape(T)[0],np.shape(B)[0]),0,dtype=float) #fictitious nodal demand
for s in range(np.shape(Omega_N)[0]):
    for t in range(np.shape(T)[0]):
        for b in range(np.shape(B)[0]):
            if (s+1 in Omega_p["C"]) or (s+1 in Omega_p["W"] and s+1 in Omega_LN_t[t+1]):
                Dtio_stb[s,t,b] = 1
            else:
                Dtio_stb[s,t,b] = 0
                
Fup_l_k = {"EFF": [3.94], #Upper limit for actual current flows through (MVA)
           "ERF": [3.94],
           "NRF": [6.28, 9],
           "NAF": [3.94, 6.28]
           }

Gup_p_k = {"C": [1, 2], #Rated capacities of generators
           "W": [0.91, 2.05]
           }

# Ref: https://wind-turbine.com/download/101655/enercon_produkt_en_06_2015.pdf
Gmax_W_sktb = np.full((np.shape(Omega_N)[0],np.shape(K_p["W"])[0],np.shape(T)[0],np.shape(B)[0]),0,dtype=float) #maximum wind power availability.
for s in range(np.shape(Omega_N)[0]): #Bus
    for k in range(np.shape(K_p["W"])[0]): #Option 
        for t in range(np.shape(T)[0]): #Stage
            for b in range(np.shape(B)[0]): #Load Level
                zone = node_zone[0,s]
                speed = wind_speed[zone-1,b]
                Gmax_W_sktb[s,k,t,b] = power_out(k+1,speed)

Gup_tr_k = {"ET": [7.5], #Upper limit for current injections of transformers.
            "NT": [12, 15]
            }

Vbase = 20 #kV
V_ = 0.95*Vbase #Lower bound for nodal voltages
Vup = 1.05*Vbase #Upper bound for nodal voltages
V_SS = 1.05*Vbase #Voltage at the substations

l__sr = np.full((np.shape(Omega_N)[0],np.shape(Omega_N)[0]),0,dtype=float) #Feeder length.
for b in branch:
    s, r = b[0]
    l__sr[s-1,r-1] = b[1]
    l__sr[r-1,s-1] = b[1]

n__DG = np.add.reduce([np.shape(Omega_p[p]) for p in P])[0] #Number of candidate nodes for installation of distributed generation

n__T = np.shape(T)[0] #number of time stages

pf = 0.9 #System power factor

H = Vup - V_  #Ref: DOI: 10.1109/TPWRS.2017.2764331

# =============================================================================
# Assets Data
# =============================================================================


i = 7.1/100 #Annual interest rate.

IB__t = [6000000, 6000000, 6000000] #Investment budget for stage t

Eta_l = {"NRF": 25, #Lifetimes of feeders in year
         "NAF": 25
        }

Eta_NT = 15 #Lifetime of new transformers

Eta_p = {"C": 20, #Lifetime of generators
         "W": 20
         }

Eta_SS = 100 #Lifetime of substations

RR_l = {"NRF": (i*(1+i)**Eta_l["NRF"])/((1+i)**Eta_l["NRF"] - 1), #Capital recovery rates for investment in feeders
        "NAF": (i*(1+i)**Eta_l["NAF"])/((1+i)**Eta_l["NAF"] - 1) 
        }

RR_NT = (i*(1+i)**Eta_NT)/((1+i)**Eta_NT - 1) #Capital recovery rates for investment in new transformers

RR_p =  {"C": (i*(1+i)**Eta_p["C"])/((1+i)**Eta_p["C"] - 1), #Capital recovery rates for investment in generators
         "W": (i*(1+i)**Eta_p["W"])/((1+i)**Eta_p["W"] - 1)
         }


RR_SS = i #Capital recovery rates for investment in substations.

Z_l_k = {"EFF": [0.732], #Unitary impedance magnitude of feeders
         "ERF": [0.732],
         "NRF": [0.557, 0.478],
         "NAF": [0.732, 0.557] 
         }

Z_tr_k = {"ET": [0.25], #impedance magnitude of transformers
          "NT": [0.16, 0.13]
          }

Delta__b = [2000, 5760, 1000] #Duration of load level b

Mi__b = load_factor #Loading factor of load level b

Vare = 0.25 #Penetration limit for distributed generation.

# =============================================================================
# Piecewise Linearization
# =============================================================================

n__V = 3 #number of blocks of the piecewise linear energy losses

M_l_kV = {"EFF": [[]], #Slope of block V of the piecewise linear energy losses for feeders
          "ERF": [[]],
          "NRF": [[], []],
          "NAF": [[], []]
          }


A_l_kV = {"EFF": [[]], #Width of block V of the piecewise linear energy losses for feeders
          "ERF": [[]],
          "NRF": [[], []],
          "NAF": [[], []]
          }

for l in L:
    for k in K_l[l]:
        for V in range(1,n__V+1,1):
            M_l_kV[l][k-1].append((2*V - 1)*Z_l_k[l][k-1]*Fup_l_k[l][k-1]/(n__V*(Vbase**2)))
            A_l_kV[l][k-1].append(Fup_l_k[l][k-1]/n__V)


M_tr_kV = {"ET": [[]], #Slope of block V of the piecewise linear energy losses for transformers
           "NT": [[],[]]
           }

A_tr_kV = {"ET": [[]], #Width of block V of the piecewise linear energy losses for transformers
           "NT": [[],[]]
           }

for tr in TR:
    for k in K_tr[tr]:
        for V in range(1,n__V+1,1):
            M_tr_kV[tr][k-1].append((2*V - 1)*Z_tr_k[tr][k-1]*Gup_tr_k[tr][k-1]/(n__V*(V_SS**2)))
            A_tr_kV[tr][k-1].append(Gup_tr_k[tr][k-1]/n__V)


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

                    + sum(sum(sum(sum(M_l_kV[l][k-1][z-1]*l__sr[s-1,r-1]*(model.delta_l_srktbv[l,s,r,k,t,b,z] - model.delta_l_srktbv[l,r,s,k,t,b,z])
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
                    model.eq5_aux1.add(model.g_tr_sktb[tr,s,k,t,b] == sum(model.delta_tr_sktbv[tr,s,k,t,b,V]
            for V in range(1,n__V+1)))

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
                for t in range(1,2): #Time stage
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
                for t in range(1,2): #Time stage 
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











