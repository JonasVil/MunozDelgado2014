"""
Created on Tue Jul 13 2022

Data of 54 bus test case used on Distribution Network Expansion Planning With an Explicit Formulation for Reliability Assessment proposed by Muñoz-Delgado et al. (2018).

Reference:
Muñoz-Delgado, G., Contreras, J., & Arroyo, J. M. (2018). Distribution Network Expansion Planning With an Explicit Formulation for Reliability Assessment. IEEE Transactions on Power Systems, 33(3), 2583 - 2596.
DOI: 10.1109/TPWRS.2017.2764331

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

n_bus = 54  #Number of buses
n_branches = 33 #Number of branches

load_factor = [0.7, 0.83, 1]

#EFF = Existing Fixed Feeder
#ERF = Existing Replaceable Feeder
#NRF = New Replacement Feeder
#NAF = New Added Feeder

branch = [ #(s,r) length  type (km)
            ((1,2	),0.66,	"EFF"),
            ((1,9	),0.86,	"ERF"),
            ((1,51	),1.11,	"ERF"),
            ((3,4	),0.90,	"ERF"),
            ((3,51	),2.06,	"ERF"),
            ((4,5	),1.45,	"EFF"),
            ((4,7	),1.24,	"EFF"),
            ((5,6	),0.81,	"EFF"),
            ((6,28	),1.55,	"NAF"),
            ((7,8	),1.00,	"EFF"),
            ((8,25	),0.79,	"NAF"),
            ((8,27	),1.60,	"NAF"),
            ((8,33	),1.92,	"NAF"),
            ((9,17	),1.61,	"NAF"),
            ((9,22	),2.08,	"NAF"),
            ((9,23	),1.36,	"EFF"),
            ((10,23	),1.89,	"EFF"),
            ((10,31	),0.92,	"NAF"),
            ((11,12	),1.42,	"ERF"),
            ((11,52	),1.50,	"ERF"),
            ((12,13	),1.70,	"EFF"),
            ((12,45	),1.33,	"NAF"),
            ((13,43	),1.07,	"NAF"),
            ((14,15	),1.81,	"ERF"),
            ((14,46	),1.31,	"NAF"),
            ((14,50	),2.25,	"NAF"),
            ((14,52	),2.21,	"ERF"),
            ((15,16	),0.91,	"EFF"),
            ((16,40	),1.29,	"NAF"),
            ((17,18	),1.83,	"NAF"),
            ((18,19	),0.68,	"NAF"),
            ((18,21	),0.98,	"NAF"),
            ((19,20	),0.96,	"NAF"),
            ((21,54	),0.58,	"NAF"),
            ((22,23	),1.85,	"NAF"),
            ((22,54	),1.89,	"NAF"),
            ((23,24	),0.82,	"NAF"),
            ((24,25	),0.89,	"NAF"),
            ((26,27	),0.68,	"NAF"),
            ((27,28	),1.15,	"NAF"),
            ((28,53	),1.64,	"NAF"),
            ((29,30	),1.17,	"NAF"),
            ((30,43	),1.47,	"NAF"),
            ((30,54	),1.02,	"NAF"),
            ((31,37	),0.45,	"NAF"),
            ((32,39	),1.46,	"NAF"),
            ((33,34	),0.81,	"NAF"),
            ((33,39	),1.19,	"NAF"),
            ((34,35	),0.76,	"NAF"),
            ((35,36	),0.45,	"NAF"),
            ((36,53	),1.28,	"NAF"),
            ((37,43	),1.01,	"NAF"),
            ((38,39	),1.19,	"NAF"),
            ((38,44	),1.27,	"NAF"),
            ((40,41	),1.39,	"NAF"),
            ((41,42	),1.52,	"NAF"),
            ((41,53	),1.73,	"NAF"),
            ((42,47	),1.82,	"NAF"),
            ((42,48	),1.77,	"NAF"),
            ((44,45	),1.02,	"NAF"),
            ((46,47	),1.29,	"NAF"),
            ((48,49	),1.58,	"NAF"),
            ((49,50	),0.92,	"NAF")
            ]

peak_demand = np.array([#Stages
                        #1          #2      #3      #4      #5      #6      #7      #8      #9      #10
                        [410.00,	422.30,	434.60,	446.90,	459.20,	471.50,	483.80,	496.10,	508.40,	520.70 ],
                        [156.00,	160.68,	165.36,	170.04,	174.72,	179.40,	184.08,	188.76,	193.44,	198.12 ],
                        [316.00,	325.48,	334.96,	344.44,	353.92,	363.40,	372.88,	382.36,	391.84,	401.32 ],
                        [64.00,		65.92,	67.84,	69.76,	71.68,	73.60,	75.52,	77.44,	79.36,	81.28  ],
                        [56.00,		57.68,	59.36,	61.04,	62.72,	64.40,	66.08,	67.76,	69.44,	71.12  ],
                        [234.00,	241.02,	248.04,	255.06,	262.08,	269.10,	276.12,	283.14,	290.16,	297.18 ],
                        [248.00,	255.44,	262.88,	270.32,	277.76,	285.20,	292.64,	300.08,	307.52,	314.96 ],
                        [144.00,	148.32,	152.64,	156.96,	161.28,	165.60,	169.92,	174.24,	178.56,	182.88 ],
                        [228.00,	234.84,	241.68,	248.52,	255.36,	262.20,	269.04,	275.88,	282.72,	289.56 ],
                        [312.00,	321.36,	330.72,	340.08,	349.44,	358.80,	368.16,	377.52,	386.88,	396.24 ],
                        [382.00,	393.46,	404.92,	416.38,	427.84,	439.30,	450.76,	462.22,	473.68,	485.14 ],
                        [186.00,	191.58,	197.16,	202.74,	208.32,	213.90,	219.48,	225.06,	230.64,	236.22 ],
                        [230.00,	236.90,	243.80,	250.70,	257.60,	264.50,	271.40,	278.30,	285.20,	292.10 ],
                        [270.00,	278.10,	286.20,	294.30,	302.40,	310.50,	318.60,	326.70,	334.80,	342.90 ],
                        [324.00,	333.72,	343.44,	353.16,	362.88,	372.60,	382.32,	392.04,	401.76,	411.48 ],
                        [432.00,	444.96,	457.92,	470.88,	483.84,	496.80,	509.76,	522.72,	535.68,	548.64 ],
                        [280.00,	288.40,	296.80,	305.20,	313.60,	322.00,	330.40,	338.80,	347.20,	355.60 ],
                        [420.00,	432.60,	445.20,	457.80,	470.40,	483.00,	495.60,	508.20,	520.80,	533.40 ],
                        [362.00,	372.86,	383.72,	394.58,	405.44,	416.30,	427.16,	438.02,	448.88,	459.74 ],
                        [0.00,		258.00,	265.74,	273.48,	281.22,	288.96,	296.70,	304.44,	312.18,	319.92 ],
                        [0.00,		32.00,	32.96,	33.92,	34.88,	35.84,	36.80,	37.76,	38.72,	39.68  ],
                        [0.00,		326.00,	335.78,	345.56,	355.34,	365.12,	374.90,	384.68,	394.46,	404.24 ],
                        [0.00,		0.00,	68.00,	70.04,	72.08,	74.12,	76.16,	78.20,	80.24,	82.28  ],
                        [0.00,		0.00,	502.00,	517.06,	532.12,	547.18,	562.24,	577.30,	592.36,	607.42 ],
                        [0.00,		0.00,	344.00,	354.32,	364.64,	374.96,	385.28,	395.60,	405.92,	416.24 ],
                        [0.00,		0.00,	0.00,	286.00,	294.58,	303.16,	311.74,	320.32,	328.90,	337.48 ],
                        [0.00,		0.00,	0.00,	326.00,	335.78,	345.56,	355.34,	365.12,	374.90,	384.68 ],
                        [0.00,		0.00,	0.00,	244.00,	251.32,	258.64,	265.96,	273.28,	280.60,	287.92 ],
                        [0.00,		0.00,	0.00,	0.00,	32.00,	32.96,	33.92,	34.88,	35.84,	36.80  ],
                        [0.00,		0.00,	0.00,	0.00,	270.00,	278.10,	286.20,	294.30,	302.40,	310.50 ],
                        [0.00,		0.00,	0.00,	0.00,	358.00,	368.74,	379.48,	390.22,	400.96,	411.70 ],
                        [0.00,		0.00,	0.00,	0.00,	46.00,	47.38,	48.76,	50.14,	51.52,	52.90  ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	294.00,	302.82,	311.64,	320.46,	329.28 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	334.00,	344.02,	354.04,	364.06,	374.08 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	414.00,	426.42,	438.84,	451.26,	463.68 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	246.00,	253.38,	260.76,	268.14,	275.52 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	164.00,	168.92,	173.84,	178.76 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	342.00,	352.26,	362.52,	372.78 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	48.00,	49.44,	50.88,	52.32  ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	188.00,	193.64,	199.28 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	260.00,	267.80,	275.60 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	348.00,	358.44,	368.88 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	218.00,	224.54,	231.08 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	88.00,	90.64  ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	224.00,	230.72 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	260.00,	267.80 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	40.00,	41.20  ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	206.00 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	234.00 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	110.00 ],
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00   ], #add eq14 problem 
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00   ], #add eq14 problem 
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00   ], #add eq14 problem 
                        [0.00,		0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00   ]  #add eq14 problem                       
                        ])

peak_demand = peak_demand/1000
# =============================================================================
# #Zones A = 1, B = 2, C = 3
#                #Buses= 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# node_zone = np.array([[2, 3, 3, 1, 1, 2, 3, 3, 1, 2, 3, 3, 3, 2, 2, 2, 3, 1, 3, 1, 3, 3, 3, 1]])
# 
# wind_speed = np.array([#Load Level (m/s)
#                        #1    2     3   
#                       [8.53, 9.12, 10.04], #Zone A
#                       [6.13, 7.26, 7.11],  #Zone B
#                       [4.13, 5.10, 5.56]   #Zone C
#                       ])
# =============================================================================

# =============================================================================
# Sets of Indexes
# =============================================================================

B = np.arange(1, len(load_factor)+1, dtype=int) #Set of Load Levels 
T = np.arange(1, np.shape(peak_demand)[1]+1, dtype=int) #Set of Time Stages
L = ["EFF", "ERF", "NRF", "NAF"]                        #Set of Feeder Types
#C = Conventional 
#W = Wind Generation 
P = ["C"]                                          #Set of Generator Types
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

K_p = {"C": [1, 2] #Sets of available alternatives for generators
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

Omega_SS = [51, 52, 53, 54] #Sets of nodes connected to node s by substation nodes
Omega_SSE = [51, 52] # Fixing eq14
Omega_SSN = [53, 54] # Fixing eq14

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
              10: [indx+1 for indx,value in enumerate(peak_demand[:, 9]) if value > 0]
              }

Omega_N = np.arange(1, n_bus+1, dtype=int) #Sets of nodes connected to node s by system nodes
Omega_p = {"C": [1, 3, 6, 8, 9, 12, 13, 17, 18, 20, 21, 23, 28, 30, 33, 35, 36, 37, 39, 43, 46, 49] #Sets of nodes connected to node s by distributed generation
           #"W": [1, 4, 5, 9, 15, 17, 18, 19]
           }


# =============================================================================
# Energy Costs
# =============================================================================

#Load Levels
#         1     2   3
C_SS_b = {51: [26.1, 38.0, 47.5], #the costs of the energy supplied by all substations
          52: [27.4, 40.0, 50.0],
          53: [28.8, 42.0, 52.5],
          54: [28.1, 41.2, 51.3]
          }

C_SS_l = [27.6, 40.3, 50.3] #Cost coefficient for losses
#DG units
C_Ep_k = {"C": [47, 45] #Conventional DG
          #"W": [0, 0]    #Windy DG
          }

#Cost for unserved energy 
C_U = 2000000

# =============================================================================
# Investment Costs
# =============================================================================

C_Il_k = {"NRF": [19140, 29870], #Investment cost coefficients of feeders
          "NAF": [15020, 25030]
          }

C_INT_k = [500000, 950000] #Investment cost coefficients of new transformers

C_Ip_k = {"C": [500000, 490000]  #Investment cost coefficients of generators
          #"W": [1850000, 1840000]
          }

C_ISS_s = {51: 100000,  #Investment cost coefficients of substations
         52: 100000, 
         53: 300000, 
         54: 300000
         }

# =============================================================================
# Maintenance Costs
# =============================================================================

C_Ml_k = {"EFF": [400], #Maintenance cost coefficients of feeders
          "ERF": [400],
          "NRF": [570, 750],
          "NAF": [400, 570]
          }

C_Mp_k = {"C": [0.05*0.9*500000*1, 0.05*0.9*490000*2] #Maintenance cost coefficients of generators
          #"W": [0.05*0.9*1850000*0.91, 0.05*0.9*1840000*2.05]
          }

C_Mtr_k = {"ET": [2000], #Maintenance cost coefficients of transformers
           "NT": [1000, 3000]
           }


# =============================================================================
# System's Data
# =============================================================================

D__st = peak_demand #Actual nodal peak demand

Dtio_stb = np.full((np.shape(Omega_N)[0],np.shape(T)[0],np.shape(B)[0]),0,dtype=float) #fictitious nodal demand
for s in range(np.shape(Omega_N)[0]):
    for t in range(np.shape(T)[0]):
        for b in range(np.shape(B)[0]):
            if (s+1 in Omega_p["C"]) and s+1 in Omega_LN_t[t+1]:
                Dtio_stb[s,t,b] = 1
            else:
                Dtio_stb[s,t,b] = 0
                
Fup_l_k = {"EFF": [6.28], #Upper limit for actual current flows through (MVA)
           "ERF": [6.28],
           "NRF": [9, 12],
           "NAF": [6.28, 9]
           }

Gup_p_k = {"C": [1, 2] #Rated capacities of generators
           #"W": [0.91, 2.05]
           }

# =============================================================================
# # Ref: https://wind-turbine.com/download/101655/enercon_produkt_en_06_2015.pdf
# Gmax_W_sktb = np.full((np.shape(Omega_N)[0],np.shape(K_p["W"])[0],np.shape(T)[0],np.shape(B)[0]),0,dtype=float) #maximum wind power availability.
# for s in range(np.shape(Omega_N)[0]): #Bus
#     for k in range(np.shape(K_p["W"])[0]): #Option 
#         for t in range(np.shape(T)[0]): #Stage
#             for b in range(np.shape(B)[0]): #Load Level
#                 zone = node_zone[0,s]
#                 speed = wind_speed[zone-1,b]
#                 Gmax_W_sktb[s,k,t,b] = power_out(k+1,speed)
# =============================================================================

Gup_tr_k = {"ET": [12], #Upper limit for current injections of transformers.
            "NT": [7.5, 15]
            }

Vbase = 13.5 #kV
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

i = 10/100 #Annual interest rate.

IB__t = [2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000] #Investment budget for stage t

Eta_l = {"NRF": 25, #Lifetimes of feeders in year
         "NAF": 25
        }

Eta_NT = 15 #Lifetime of new transformers

Eta_p = {"C": 20 #Lifetime of generators
         #"W": 20
         }

Eta_SS = 100 #Lifetime of substations

RR_l = {"NRF": (i*(1+i)**Eta_l["NRF"])/((1+i)**Eta_l["NRF"] - 1), #Capital recovery rates for investment in feeders
        "NAF": (i*(1+i)**Eta_l["NAF"])/((1+i)**Eta_l["NAF"] - 1) 
        }

RR_NT = (i*(1+i)**Eta_NT)/((1+i)**Eta_NT - 1) #Capital recovery rates for investment in new transformers

RR_p =  {"C": (i*(1+i)**Eta_p["C"])/((1+i)**Eta_p["C"] - 1) #Capital recovery rates for investment in generators
         #"W": (i*(1+i)**Eta_p["W"])/((1+i)**Eta_p["W"] - 1)
         }


RR_SS = (i*(1+i)**Eta_SS)/((1+i)**Eta_SS - 1) #Capital recovery rates for investment in substations.

Z_l_k = {"EFF": [0.557], #Unitary impedance magnitude of feeders
         "ERF": [0.557],
         "NRF": [0.478, 0.423],
         "NAF": [0.557, 0.478] 
         }

Z_tr_k = {"ET": [0.16], #impedance magnitude of transformers
          "NT": [0.25, 0.13]
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
