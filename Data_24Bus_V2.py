"""
Created on Mon Nov 30 13:32:38 2020

Data of 24 bus test case used on Distribution Expansion Planning Model proposed by Muñoz-Delgado et al. (2014).

Reference:
Muñoz-Delgado, G., Contreras, J., & Arroyo, J. M. (2014). Joint expansion planning of distributed generation and distribution networks. IEEE Transactions on Power Systems, 30(5), 2579-2590.
DOI: 10.1109/TPWRS.2014.2364960

@Code Athor: Jonas Villela de Souza
"""
import numpy as np

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

Upsilon_N = {"EFF": [],
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
            if s not in Upsilon_N[branch_type]:
                Upsilon_N[branch_type].append(s)
            if r not in Upsilon_N[branch_type]:
                Upsilon_N[branch_type].append(r)
Upsilon_l["NRF"] = Upsilon_l["ERF"]
Upsilon_N["NRF"] = Upsilon_N["ERF"]

# =============================================================================
# Sets of Nodes
# =============================================================================

Omega_SS = [21, 22, 23, 24] #Sets of nodes connected to node s by substation nodes
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

Omega_N = np.arange(1, n_bus+1-4, dtype=int) #Sets of nodes connected to node s by system nodes
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
            if (s+1 in Omega_p["C"] or s+1 in Omega_p["W"]) and s+1 in Omega_LN_t[t+1]:
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

l__sr = np.full((np.concatenate((Omega_N, Omega_SS), axis=None).shape[0],np.concatenate((Omega_N, Omega_SS), axis=None).shape[0]),0,dtype=float) #Feeder length.
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


RR_SS = (i*(1+i)**Eta_SS)/((1+i)**Eta_SS - 1) #Capital recovery rates for investment in substations.

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

#Vare = 0.25 #Penetration limit for distributed generation.

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
