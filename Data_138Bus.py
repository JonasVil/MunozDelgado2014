# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:53:07 2020

Data of 138 bus test case used on Distribution Expansion Planning Model proposed by Muñoz-Delgado et al. (2014).

Reference:
Muñoz-Delgado, G., Contreras, J., & Arroyo, J. M. (2014). Joint expansion planning of distributed generation and distribution networks. IEEE Transactions on Power Systems, 30(5), 2579-2590.
DOI: 10.1109/TPWRS.2014.2364960

@Code Athor: Jonas Villela de Souza
"""
import numpy as np
import pandas as pd

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

n_bus = 138  #Number of buses
n_branches = 151 #Number of branches

load_factor = [0.7, 0.83, 1]

#EFF = Existing Fixed Feeder
#ERF = Existing Replaceable Feeder
#NRF = New Replacement Feeder
#NAF = New Added Feeder

line_data = pd.read_csv("138_line_data.csv")
branch = []
for i in range(line_data.shape[0]):
    if line_data['From'][i] == 201:
        s = 136
    elif line_data['From'][i] == 202:
        s = 137
    elif line_data['From'][i] == 203:
        s = 138
    else:
        s = line_data['From'][i]
    r = line_data['to'][i]
    l = line_data['Lenght'][i]
    tYpe = line_data['Type'][i]
    branch.append(((s,r), l, tYpe))

load_zone = pd.read_csv("138_load_zone.csv")

peak_demand = np.full((load_zone.shape[0],10),0,dtype=float)
for i in range(0,load_zone.shape[0]):
    for j in range(1,10+1):
        peak_demand[i,j-1] = load_zone[str(j)][i]

#Zones A = 1, B = 2, C = 3
               #Buses= 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 ... 138   
node_zone = np.full((1,load_zone.shape[0]),0,dtype=int) 
for i in range(0,load_zone.shape[0]):
    if load_zone['Zone'][i] == 'A':
        node_zone[0,i] = 1 
    elif load_zone['Zone'][i] == 'B':
        node_zone[0,i] = 2 
    elif load_zone['Zone'][i] == 'C':
        node_zone[0,i] = 3
    
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

Omega_SS = [136, 137, 138] #Sets of nodes connected to node s by substation nodes
Omega_SSE = [136, 137] # Fixing eq14
Omega_SSN = [138] # Fixing eq14

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
              3: [indx+1 for indx,value in enumerate(peak_demand[:, 2]) if value > 0],
              4: [indx+1 for indx,value in enumerate(peak_demand[:, 3]) if value > 0],
              5: [indx+1 for indx,value in enumerate(peak_demand[:, 4]) if value > 0],
              6: [indx+1 for indx,value in enumerate(peak_demand[:, 5]) if value > 0],
              7: [indx+1 for indx,value in enumerate(peak_demand[:, 6]) if value > 0],
              8: [indx+1 for indx,value in enumerate(peak_demand[:, 7]) if value > 0],              
              9: [indx+1 for indx,value in enumerate(peak_demand[:, 8]) if value > 0],
              10: [indx+1 for indx,value in enumerate(peak_demand[:, 9]) if value > 0],                
              }

Omega_N = np.arange(1, n_bus+1, dtype=int) #Sets of nodes connected to node s by system nodes

Omega_p = {"C": [10, 28, 38, 53, 64, 94, 108, 117, 126, 133], #Sets of nodes connected to node s by distributed generation
           "W": [31, 52, 78, 94, 103, 113, 114, 116, 120, 122]
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

C_Il_k = {"NRF": [29870, 39310], #Investment cost coefficients of feeders
          "NAF": [25030, 34920]
          }

C_INT_k = [500000, 950000] #Investment cost coefficients of new transformers

C_Ip_k = {"C": [500000, 490000],  #Investment cost coefficients of generators
          "W": [1850000, 1840000]
          }

C_ISS_s = {136: 100000,  #Investment cost coefficients of substations
         137: 100000, 
         138: 150000
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
                
Fup_l_k = {"EFF": [6.28], #Upper limit for actual current flows through (MVA)
           "ERF": [6.28],
           "NRF": [9.00, 12],
           "NAF": [6.28, 9]
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

Gup_tr_k = {"ET": [12], #Upper limit for current injections of transformers.
            "NT": [7.5, 15]
            }




















    
    
    