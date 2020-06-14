#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
from scipy import optimize


# In[2]:


np.random.seed(seed=42)


# In[5]:


age_and_sex = pd.read_csv('age_and_sex.csv').drop(['Unnamed: 0'], axis = 1)


# In[7]:


age_and_sex.values


# In[ ]:


class Tucker_Model():
    def __init__(self,
                 num_ppl_iso, 
                 mean_iso, 
                 prop_iso, 
                 num_ppl_tents, 
                 max_tent,
                 prob_house,
                 prob_meet,
                 init_trans,
                 prob_symp,
                 recovery,
                 perm_asymp,
                 int_eth,
                 small_rad,
                 large_rad,
                 rad_int,
                 filename='age_and_sex.csv'):
        
        self.num_ppl_iso = num_ppl_iso # Number of people in isoboxes. (Nb)
        self.mean_iso = mean_iso # Isoboxes mean occupancy (people). (mub)
        self.num_iso = num_ppl_iso/mean_iso # Number of isoboxes. (hb)
        self.prop_iso = prop_iso # Proportion of area covered by isoboxes. (iba)
        self.num_ppl_tents = num_ppl_tents # Number of people in tents. (Nt)
        self.max_tent = max_tent # Tents occupancy of (people). (mut)
        self.num_tents = num_ppl_tents/max_tent # Number of tents. (ht)
        self.total_pop = num_ppl_tents + num_ppl_iso # Total population. (N)
        self.prob_house = prob_house # Probability of infecting each person in your household per day. (twh)
        self.prob_meet = prob_meet # Probability of infecting each person you meet per meeting (Fang et al.) (aip)
        self.init_trans = init_trans # Initial transmission reduction (relative to assumed per contact transmission rate, outside household only). (tr)
        self.prob_symp = prob_symp # Probability of spotting symptoms, per person per day. (siprob)
        self.recovery = recovery # Days in quarantine after no virus shedding (i.e., recovery). (clearday)
        self.perm_asymp = perm_asymp # Permanently asymptomatic cases (Mizumoto et al 2020 Eurosurveillance). (pac)
        self.int_eth = int_eth # Realtive strength of interaction between different ethnicities. (ss)
        self.small_rad = small_rad # Smaller movement radius. Range around their household during lockdown or females and individuals age < 10. (lr1)
        self.large_rad = large_rad # Larger movement radius. ie. Pople who violate lockdown enforcement or males over age 10. (lr2)
        self.rad_int = rad_int # Scale interactions - two people with completely overlapping rages with this radius interact once per day (lrtol)
        self.data = pd.read_csv(filename).drop(['Unnamed: 0'], axis = 1).values
        
    def create_matrix(self):
        """
        Creates N x 9 matrix, with the following column definitions:
        
        Columns:
        0. Home number
        1. Disease state: 0 = susceptible, 1 = exposed, 2 = presymptomatic, 3 = symptomatic, 4 = mild, 5 = severe, 
        6 = recovered. Similar states in quarentine are the same plus seven. 
        In other words, this is a categorical variable with values between 0 and 13 inclusive.
        2. Days to symptoms for this person
        3. Days passed in current state
        4. Whether this person will be asymptomatic
        5. Age
        6. Male: = 1 if male.
        7. Chronic: = 1 if chronic disease?
        8. Wanderer (Uses the larger radius)
        
        This is the primary object that is operated on within the class. It will model the epidemic within the community
        by adjusting the parameters in the weekend.
        """
        
        column_0 = np.subtract(np.concatenate((np.ceil(self.num_iso*np.random.uniform(0,1,self.num_ppl_iso)), 
                                               self.num_iso+np.ceil(self.num_tents*np.random.uniform(0,1,self.num_ppl_tents)))),
                               1).astype(int)
        column_1 = np.zeros(np.shape(column_1))
        
        #Columns 2-4 require additional parameters as described in (Backer et al. 2020 Eurosurveillance).
        #Primarily influence column 2
        
        k = (2.3/6.4)**(-1.086)
        L = 6.4 / (math.gamma(1 + 1/k))
        column_2 = k*np.random.weibull(L,(self.total_pop,1))
        column_3 = np.zeros((self.total_pop,1))
        column_4 = (np.random.uniform(0,1,self.total_pop)<pac*(self.total_pop/(self.total_pop-300))).astype(int).astype(float)
        column_5 = self.data[np.random.randint(np.shape(self.data)[0], size=self.total_pop)][:, 0]
        column_6 = self.data[np.random.randint(np.shape(self.data)[0], size=self.total_pop)][:, 1]
        
        #For this portion, we need to optimize age to chronic states
        myfunction = lambda x: np.absolute(300-np.sum((1+np.exp(-(x-11.69+.2191*column_5-0.001461*column_5**2))**(-1))))-self.total_pop
        xopt = optimize.fsolve(myfunction, x0=[2])
        rchron = (1+np.exp(-(xopt-11.69+.2191*column_5-0.001461*column_5**2)))**(-1)
        chroncases = (np.random.uniform(np.min(rchron),1,self.total_pop) < rchron).astype(int).astype(float)
        
        #Update the number of asymptomatic cases based on the number of chronic cases
        column_4[np.where(chroncases == 1)] = 0
        column_7 = chroncases
        column_8 = np.logical_and([column_6 == 1], [10 <= column_5]).transpose().astype(int).astype(float)
        
        matrix = np.column_stack((column_0, column_1, column_2, column_3, column_4, column_5, column_6, column_7, column_8))
        
        return matrix
    
    def assignBlock(self, hhlocc, maxhh, blocks):
        """
        Out
        num: Assign a block to a household (ie. toilet block or food line)
        shared: A Matrix of shared blocks at the household level.
        """
        grid1 = np.sum(np.outer(hhloc[:,0],np.ones((1,blocks[0])))>np.outer(np.ones((maxhh,1)),np.arange(1,blocks[0]+1)/blocks[0]),1)
        grid2 = np.sum(np.outer(hhloc[:,1],np.ones((1,blocks[1])))>np.outer(np.ones((maxhh,1)),np.arange(1,blocks[1]+1)/blocks[1]),1)
        num = grid2*blocks[0]+grid1+1
        TEMP = np.outer(num,np.ones((1,len(num))))
        shared = (TEMP.T == TEMP) - np.eye(maxhh)
        return [num,shared]

    def create_households(self, nx=12, ny=12):
        #Create model matrix
        matrix = self.create_matrix()
        
        #Create parameters for people per household
        pph = np.bincount(matrix[:,0])
        maxhh = pph.size
        
        #Create coordinates for households
        hhloc1 = 0.5*(1-np.sqrt(self.prop_iso)) + np.sqrt(self.prop_iso)*np.random.uniform(0,1,(int(self.num_iso),2))
        hhloc2 = np.random.uniform(0,1,(int(matrix[self.total_pop-1,0]-matrix[self.num_ppl_iso-1,0]),2))
        hhloc2w = np.ceil(4*np.random.uniform(0,1,(int(matrix[self.total_pop-1,0]-matrix[self.num_ppl_iso-1,0]),1)))
        shift = 0.5*(1-np.sqrt(self.prop_iso))
        hhloc2[np.where(hhloc2w==1),0] = 1+shift*(hhloc2[np.where(hhloc2w == 1),0]-1)
        hhloc2[np.where(hhloc2w==1),1] = (1-shift)*hhloc2[np.where(hhloc2w == 1),1]
        hhloc2[np.where(hhloc2w==2),0] = hhloc2[np.where(hhloc2w == 2),0]*(1-shift)+shift
        hhloc2[np.where(hhloc2w==2),1] = 1+shift*(hhloc2[np.where(hhloc2w == 2),1]-1)
        hhloc2[np.where(hhloc2w==3),0] = shift*hhloc2[np.where(hhloc2w == 3),0]
        hhloc2[np.where(hhloc2w==3),1] = hhloc2[np.where(hhloc2w == 3),1]*(1-shift)+shift
        hhloc2[np.where(hhloc2w==4),0] = (1-shift)*hhloc2[np.where(hhloc2w == 4),0]
        hhloc2[np.where(hhloc2w==4),1] = shift*hhloc2[np.where(hhloc2w == 4),1]
        hhloc = np.vstack((hhloc1,hhloc2)) 
        
        #Initialize grid parameters
        tblocks = np.array([nx,ny])       # Grid dimensions.
        tgroups = tblocks[0]*tblocks[1]   # Number of blocks in the grid.
        tu = N/tgroups                    # ~ people / toilet
        [tnum,tshared] = assignBlock(hhloc,maxhh,tblocks)
        
        return hhloc, maxhh
    
    def ethnicities(self, ethnicities, totEthnic, maxhh):
        num_hhpg = np.round(maxhh*ethnicities/totEthnic)
        num_hhpg = np.random.choice(ethnicities,len(ethnicities),replace=False)
        hhunass = np.column_stack((np.arange(0,maxhh), hhloc))
        hheth = np.zeros((maxhh,1))
        
        for i in range(len(num_hhpg)):
            gcen = hhunass[np.random.randint(hhunass.shape[0]),1:] # Chose an unassigned household as the group (cluster) centre.
            dfromc = np.sum((hhunass[:,1:]-np.tile(gcen,(hhunass.shape[0],1)))**2,1) # Squared distance to cluster centre.
            cloind = np.argsort(dfromc)                            # Get the indices of the closest households (cloind).
            hheth[hhunass[cloind[0:int(num_hhpg[i])],0].astype(int)] = i  # Assign i-th ethnic group to those households.
            hhunass = np.delete(hhunass,cloind[0:int(g[i])],0)     # Remove those hoseholds (remove the i-th cluster/ethnic group)
        
        ethmatch = ( np.tile(hheth,(1,len(hheth))) ==  np.tile(hheth,(1,len(hheth))).T )
        ethcor = ethmatch+self.int_eth*(1-ethmatch)
        
        return ethcor

