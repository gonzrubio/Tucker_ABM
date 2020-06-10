import numpy as np
import random
import math
from scipy import optimize

def create_household_column(num_hh_type1,num_ppl_type1,num_hh_type2,num_ppl_type2):
    ppl_hh_index_draw= np.concatenate((np.ceil(num_hh_type1*np.random.uniform(0,1,num_ppl_type1)), 
                       num_hh_type1+np.ceil(num_hh_type2*np.random.uniform(0,1,num_ppl_type2)))) 
    hh_index,ppl_to_hh_index = np.unique(ppl_hh_index_draw, return_inverse=True)  # ui - indices from the unique sorted array that would reconstruct rN
    assert hh_index[ppl_to_hh_index].all() == ppl_hh_index_draw.all()
    return np.sort(ppl_to_hh_index)

def create_diseasestate_column(num_ppl,seed=1):
    initial_diseasestate=np.zeros(num_ppl)
    initial_diseasestate[np.random.choice(num_ppl, seed)] = 1
    return initial_diseasestate

def create_daystosymptoms_column(num_ppl):
    #weibull distribution parameters following (Backer et al. 2020 Eurosurveillance)
    k = (2.3/6.4)**(-1.086)
    L = 6.4 / (math.gamma(1 + 1/k))
    return np.array([random.weibullvariate(L,k) for ppl in np.arange(num_ppl)])

def create_daycount_column(num_ppl):
    return np.zeros(num_ppl)

def create_asymp_column(num_ppl,asymp_rate,age_column=None,num_ppl_chro=300):
    """
    num_ppl_pre: number of people with chronic diseases (pre-exisitng medical conditions) meaning they won't be asymptomatically infected
    """
    if age_column is not None:
        pass
    else:
        return np.random.uniform(0,1,num_ppl)<asymp_rate*(num_ppl/(num_ppl-num_ppl_chro))

def create_age_column(age_data):
    return age_data

def create_gender_column(gender_data):
    return gender_data

def create_chronic_column(num_ppl,age_column,num_ppl_chro=300):
    myfunction = lambda x: np.absolute(num_ppl_chro-np.sum((1+np.exp(-(x-11.69+.2191*age_column-0.001461*age_column**2))**(-1))))-num_ppl
    xopt = optimize.fsolve(myfunction, x0=[2])
    rchron = (1+np.exp(-(xopt-11.69+.2191*age_column-0.001461*age_column**2)))**(-1)
    chroncases = (np.random.uniform(np.min(rchron),1,num_ppl) < rchron)
    return chroncases

def adjust_asymp_with_chronic(asymp_column,chronic_column):
    new_asymp_column=asymp_column.copy()
    new_asymp_column[chronic_column==1]=0
    return new_asymp_column

def create_wanderer_column(gender_column,age_column):
    """
    Male of age greater than 10 are the wanderers in the camp
    """
    return np.logical_and([gender_column==1], [age_column>=10]).transpose()

def form_population_matrix(N,hb,Nb,ht,Nt,pac,age_and_gender):
    #1
    household_column=create_household_column(hb,Nb,ht,Nt)
    #2
    disease_column=create_diseasestate_column(N)
    #3
    dsymptom_column=create_daystosymptoms_column(N)
    #4
    daycount_column=create_daycount_column(N)
    #5
    asymp_column=create_asymp_column(N,pac)
    #6
    age_column=create_age_column(age_and_gender[:,0])
    #7
    gender_column=create_gender_column(age_and_gender[:,1])
    #8
    chronic_column=create_chronic_column(N,age_column)
    #5
    new_asymp_column=adjust_asymp_with_chronic(asymp_column,chronic_column)
    #9
    wanderer_column=create_wanderer_column(gender_column,age_column)
    pop_matrix=np.column_stack((household_column,disease_column,dsymptom_column,daycount_column,new_asymp_column,age_column,gender_column,chronic_column,wanderer_column))
    assert pop_matrix.shape==(N, 9)
    return pop_matrix

def place_households(ppl_to_hh_index,prop_type1,num_hh_type1):
    pph = np.bincount(ppl_to_hh_index)
    maxhh = pph.size #number of total households
    # Assign x and y coordinates to isoboxes (there are hb total isoboxes). 
    hhloc1 = 0.5*(1-np.sqrt(prop_type1)) + np.sqrt(prop_type1)*np.random.uniform(0,1,(int(num_hh_type1),2))
    # Repeat for tents.
    hhloc2 = np.random.uniform(0,1,(int(maxhh-num_hh_type1),2)) # Note: Nb-1 and N-1 to account for zero-indexing.
    assert (hhloc1.shape[0]+hhloc2.shape[0] == maxhh)
    # Randomly move tents to the edges of the camp. Assign randomly a side to each of the household.
    hhloc2w=np.random.randint(1, 5, size=int(maxhh-num_hh_type1))
    assert len(hhloc2w) == hhloc2.shape[0]
    # This block moves some tents to the right edge.
    shift = 0.5*(1-np.sqrt(prop_type1)) #this is the width of the gap assuming isobox occupies a square in the middle with half the area 
    #(interesting parameter to tune)
    hhloc2[np.where(hhloc2w==1),0] = shift*hhloc2[np.where(hhloc2w == 1),0]+(1-shift)
    hhloc2[np.where(hhloc2w==1),1] = (1-shift)*hhloc2[np.where(hhloc2w == 1),1] #shrink towards bottom right
    # This block moves some tents to the upper edge.
    hhloc2[np.where(hhloc2w==2),0] = hhloc2[np.where(hhloc2w == 2),0]*(1-shift)+shift #push towards top right
    hhloc2[np.where(hhloc2w==2),1] = shift*hhloc2[np.where(hhloc2w == 2),1]+(1-shift)
    # This block moves some tents to the left edge.
    hhloc2[np.where(hhloc2w==3),0] = shift*hhloc2[np.where(hhloc2w == 3),0]
    hhloc2[np.where(hhloc2w==3),1] = hhloc2[np.where(hhloc2w == 3),1]*(1-shift)+shift #push it towards top left
    # This block moves some tents to the bottom edge.
    hhloc2[np.where(hhloc2w==4),0] = (1-shift)*hhloc2[np.where(hhloc2w == 4),0] #push it towards bottom left
    hhloc2[np.where(hhloc2w==4),1] = shift*hhloc2[np.where(hhloc2w == 4),1] 
    hhloc = np.vstack((hhloc1,hhloc2)) 
    assert hhloc.shape[0] == maxhh
    return hhloc

def position_toilet(hhloc,nx = 12,ny = 12):
    tblocks = np.array([nx,ny])       # Grid dimensions.
    # tgroups = tblocks[0]*tblocks[1]   # Number of blocks in the grid.
    # tu = num_ppl/tgroups                    # ~ people / toilet
    hh_size=hhloc.shape[0]
    tlimit_x = np.arange(1,tblocks[0]+1)/tblocks[0]
    tlabel_x = np.digitize(hhloc[:,0], tlimit_x)
    tlimit_y = np.arange(1,tblocks[1]+1)/tblocks[1]
    tlabel_y = np.digitize(hhloc[:,1], tlimit_y)
    tlabel = tlabel_y*tblocks[0]+tlabel_x+1
    #find out which households share the same toilet
    TEMP = np.outer(tlabel,np.ones((1,len(tlabel))))
    tshared = (TEMP.T == TEMP) - np.eye(hh_size)
    assert np.max(tlabel) == np.prod(tblocks)
    assert tshared.shape == (hh_size,hh_size)
    return [tlabel,tshared]