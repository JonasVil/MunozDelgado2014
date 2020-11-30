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
    
    
    
    
    
    
    
    
    
    
    
    
    
    