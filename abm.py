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

def assign_block(hhloc,blocks):
    """
    Out
    num: Assign a block to a household (ie. toilet block or food line)
    shared: A Matrix of shared blocks at the household level.
    """
    hh_size=hhloc.shape[0]
    limit_x = np.arange(1,blocks[0]+1)/blocks[0]
    label_x = np.digitize(hhloc[:,0], limit_x)
    limit_y = np.arange(1,blocks[1]+1)/blocks[1]
    label_y = np.digitize(hhloc[:,1], limit_y)
    label = label_y*blocks[0]+label_x+1
    #find out which households share the same toilet
    TEMP = np.tile(label,(len(label),1))
    shared = (TEMP.T == TEMP) - np.eye(hh_size)
    assert np.max(label) == np.prod(blocks)
    assert shared.shape == (hh_size,hh_size)
    return label,shared

def position_toilet(hhloc,nx = 12,ny = 12):
    tblocks = np.array([nx,ny])       # Grid dimensions.
    # tgroups = tblocks[0]*tblocks[1]   # Number of blocks in the grid.
    # tu = num_ppl/tgroups                    # ~ people / toilet
    tlabel,tshared=assign_block(hhloc,tblocks)
    return [tlabel,tshared]

def position_foodline(hhloc,nx=1,ny=1):
    fblocks = np.array([nx,ny]) 
    flabel,fshared=assign_block(hhloc,fblocks)
    return flabel,fshared

def create_ethnic_groups(hhloc,int_eth):
    Afghan = 7919 ; Cameroon = 149 ; Congo = 706 ;Iran = 107 ;Iraq = 83 ; Somalia = 442 ; Syria = 729
    g = np.array([Afghan,Cameroon,Congo,Iran,Iraq,Somalia,Syria])  
    totEthnic = sum(g) 
    hh_size=hhloc.shape[0]
    g_hh = np.round(hh_size*g/totEthnic)              # Number of households per group.
    np.random.shuffle(g_hh) #shuffle the array
    hhunass= np.column_stack((np.arange(0,hh_size), hhloc))   # Unassigned households. Firsto column is the index of the hh.
    hheth = np.zeros((hh_size,1))
    i=0
    for g in g_hh:
        gcen = hhunass[np.random.randint(hhunass.shape[0]),1:] # Chose an unassigned household as the group (cluster) centre.
        dfromc = np.sum((hhunass[:,1:]-np.tile(gcen,(hhunass.shape[0],1)))**2,1) # Squared distance to cluster centre.
        cloind = np.argsort(dfromc)                            # Get the indices of the closest households (cloind).
        hheth[hhunass[cloind[0:int(g)],0].astype(int)] = i  # Assign i-th ethnic group to those households.
        hhunass = np.delete(hhunass,cloind[0:int(g)],0)     # Remove those hoseholds (remove the i-th cluster/ethnic group)
        i+=1
    ethmatch = (np.tile(hheth,(1,len(hheth)))==np.tile(hheth,(1,len(hheth))).T )
    #scale down the connection for poeple of different background
    ethcor = ethmatch+int_eth*(1-ethmatch)
    return ethcor