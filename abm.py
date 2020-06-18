import numpy as np
import pandas as pd
import random
import math
from scipy import optimize
from scipy.special import factorial

def read_age_gender(num_ppl):
    path_to_file = 'age_and_sex.csv'            # Observed data.
    age_and_gender = pd.read_csv(path_to_file)     # Data frame. V1 = age, V2 is sex (1 = male?, 0  = female?).
    age_and_gender = age_and_gender.loc[:, ~age_and_gender.columns.str.contains('^Unnamed')] 
    age_and_gender = age_and_gender.values 
    age_and_gender = age_and_gender[np.random.randint(age_and_gender.shape[0], size=num_ppl)]
    return age_and_gender

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
    return tlabel,tshared

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

def interaction_neighbours(hhloc,lr1,lr2,lrtol,ethcor):
    #create distance matrix for distance in between households
    hhdm_x=(np.tile(hhloc[:,0],(hhloc.shape[0],1)).T - np.tile(hhloc[:,0],(hhloc.shape[0],1)))**2
    hhdm_y=(np.tile(hhloc[:,1],(hhloc.shape[0],1)).T - np.tile(hhloc[:,1],(hhloc.shape[0],1)))**2
    hhdm=np.sqrt(hhdm_x+hhdm_y)
    #the case where lr1 is inteacting with lr1
    angle11=2*np.arccos(np.clip(hhdm/(2*lr1),a_min=None,a_max=1)) 
    area_sector11=0.5*(lr1**2)*angle11
    area_overlap11=area_sector11*2-lr1*np.sin(angle11/2)*hhdm
    relative_encounter11=area_overlap11/(math.pi**2*lr1**4)
    #the case where lr2 is inteacting with lr2
    angle22=2*np.arccos(np.clip(hhdm/(2*lr2),a_min=None,a_max=1)) 
    area_sector22=0.5*(lr2**2)*angle22
    area_overlap22=area_sector22*2-lr2*np.sin(angle22/2)*hhdm
    relative_encounter22=area_overlap22/(math.pi**2*lr2**4)
    #the case where lr1 is interacting with lr2
    angle1=2*np.arccos(np.clip((hhdm**2+lr2**2-lr1**2)/(2*hhdm*lr2),a_min=None,a_max=1)) #nan means no overlap in this case
    angle2=2*np.arccos(np.clip((hhdm**2+lr1**2-lr2**2)/(2*hhdm*lr1),a_min=None,a_max=1)) #nan means no overlap in this case
    area_sector1=0.5*(lr1**2)*angle2
    area_sector2=0.5*(lr2**2)*angle1
    area_overlap12=np.nan_to_num(area_sector1+area_sector2-lr1*np.sin(angle2/2)*hhdm)
    relative_encounter12=area_overlap12/(math.pi**2*lr2**2*lr1**2)
    lis = np.multiply(math.pi*lrtol**2*np.dstack((relative_encounter11,relative_encounter12,relative_encounter22)),np.dstack((ethcor,ethcor,ethcor)))
    return lis

def interaction_neighbours_fast(hhloc,lr1,lr2,lrtol,ethcor):
    #use the formula from https://mathworld.wolfram.com/Circle-CircleIntersection.html 
    #create distance matrix for distance in between households
    hhdm_x=(np.tile(hhloc[:,0],(hhloc.shape[0],1)).T - np.tile(hhloc[:,0],(hhloc.shape[0],1)))**2
    hhdm_y=(np.tile(hhloc[:,1],(hhloc.shape[0],1)).T - np.tile(hhloc[:,1],(hhloc.shape[0],1)))**2
    hhdm=np.sqrt(hhdm_x+hhdm_y)
    #the case where lr1 is inteacting with lr1
    area_overlap11=2*(lr1**2*np.arccos(np.clip(0.5*hhdm/lr1,a_min=None,a_max=1))-np.nan_to_num(hhdm/2*np.sqrt(lr1**2-hhdm**2/4)))
    relative_encounter11=area_overlap11/(math.pi**2*lr1**4)
    #the case where lr2 is inteacting with lr2
    area_overlap22=2*(lr2**2*np.arccos(np.clip(hhdm/(2*lr2),a_min=None,a_max=1))-np.nan_to_num(hhdm/2*np.sqrt(lr2**2-hhdm**2/4)))
    relative_encounter22=area_overlap22/(math.pi**2*lr2**4)
    #the case where lr1 is interacting with lr2
    area_overlap12=np.nan_to_num((lr1**2*np.arccos(np.clip((hhdm**2+lr1**2-lr2**2)/(2*hhdm*lr1),a_min=None,a_max=1)))
    +(lr2**2*np.arccos(np.clip((hhdm**2+lr2**2-lr1**2)/(2*hhdm*lr2),a_min=None,a_max=1)))
    -0.5*np.sqrt((-hhdm+lr1+lr2)*(hhdm+lr1-lr2)*(hhdm-lr1+lr2)*(hhdm+lr1+lr2)))
    relative_encounter12=area_overlap12/(math.pi**2*lr2**2*lr1**2)
    lis = np.multiply(math.pi*lrtol**2*np.dstack((relative_encounter11,relative_encounter12,relative_encounter22)),np.dstack((ethcor,ethcor,ethcor)))
    return lis

def epidemic_finish(states,iteration):
    '''
    Finish the simulation when no person is in any state other than recovered or susceptible
    '''
    return (np.sum(states) == 0 and iteration > 10)

def disease_state_update(pop_matrix,mild_rec,sev_rec,pick_sick,thosp,quarantined=False):
    #progress infected individuals to mild infection, progress mild to recovered and progress severe to recovered
    qua_add=0
    if quarantined:
        qua_add=7
    idxrecovmild = np.logical_and(pop_matrix[:,1]==(4+qua_add),mild_rec)
    idxrecovsevere = np.logical_and(pop_matrix[:,1]==(5+qua_add),sev_rec)
    pop_matrix[idxrecovmild,1] = 6+qua_add                             # Mild symptoms and recovered.
    pop_matrix[idxrecovsevere,1] = 6+qua_add                              # Severe symptoms and recovered.
    idxsympmild=np.logical_and(pop_matrix[:,1]==(3+qua_add),pop_matrix[:,3]==6)
    pop_matrix[idxsympmild,1] = 4+qua_add   # Move individuals with 6 days of symptoms to mild.
    asp = np.array([0,.000408,.0104,.0343,.0425,.0816,.118,.166,.184])        # Verity et al. hospitalisation.
    aspc = np.array([.0101,.0209,.0410,.0642,.0721,.2173,.2483,.6921,.6987])  # Verity et al. corrected for Tuite.
    AGE_BUCKET=9
    for buc in range(AGE_BUCKET):
        # Assign individuals with mild symptoms for six days, sick, between 10*sci and 10*sci+1 years old to severe and count as hospitalized.
        if buc==8:
            #here we include everyone beyond the age of 80 instead of 80-90
            severe_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<asp[buc],pop_matrix[:,5]>=10*buc))
            thosp += np.sum(severe_ind)
            pop_matrix[severe_ind,1] = 5+qua_add                 
            # Wouldnt this step double count previous individuals? Is this step the one that adjusts for pre-existing conditions?
            severe_chronic_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<aspc[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,7]==1))
            thosp += np.sum(severe_chronic_ind)
            pop_matrix[severe_chronic_ind,1] = 5+qua_add
        severe_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<asp[buc],pop_matrix[:,5]>=10*buc, pop_matrix[:,5]<(10*buc+10)))
        thosp += np.sum(severe_ind)
        pop_matrix[severe_ind,1] = 5+qua_add                
        # Wouldnt this step double count previous individuals? Is this step the one that adjusts for pre-existing conditions?
        severe_chronic_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<aspc[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,5]<(10*buc+10),pop_matrix[:,7]==1))
        thosp += np.sum(severe_chronic_ind)
        pop_matrix[severe_chronic_ind,1] = 5+qua_add
    # Move presymptomatic to symptomatic but not yet severe.
    idxpresym = np.logical_and(pop_matrix[:,1]==2+qua_add,pop_matrix[:,3]>=pop_matrix[:,2])
    pop_matrix[idxpresym,1] = 3+qua_add
    pop_matrix[idxpresym,3] = 0
    # Move to exposed to presymptomatic
    exptopre_ind=np.logical_and(pop_matrix[:,1]==1+qua_add,pop_matrix[:,3]>=np.floor(0.5*pop_matrix[:,2]))
    pop_matrix[exptopre_ind,1] =2+qua_add
    return pop_matrix,thosp

def accumarray(subs,val):
    '''Construct Array with accumulation.
    https://www.mathworks.com/help/matlab/ref/accumarray.html'''
    return np.array([np.sum(val[np.where(subs==i)]) for i in np.unique(subs)])

def identify_contagious_active(pop_matrix):
    contagious_hhl = np.logical_and(pop_matrix[:,1]>1,pop_matrix[:,1]<6)  
    contagious_hhl_qua = np.logical_and(pop_matrix[:,1]>8,pop_matrix[:,1]<13) 
    contagious_asymp=np.logical_and.reduce((pop_matrix[:,1]>2,pop_matrix[:,1]<5,pop_matrix[:,4]==1))
    contagious_presymp=(pop_matrix[:,1]==2)
    contagious_teen=np.logical_and.reduce((pop_matrix[:,1]>2,pop_matrix[:,1]<5,pop_matrix[:,5]<16))
    contagious_camp=np.logical_or.reduce((contagious_asymp,contagious_presymp,contagious_teen))
    contagious_sitters = np.logical_and(contagious_camp,pop_matrix[:,8]==False)
    contagious_wanderers = np.logical_and(contagious_camp,pop_matrix[:,8]==True)
    active_camp=np.logical_or.reduce((contagious_camp,pop_matrix[:,1]<2,pop_matrix[:,1]==6))
    assert(sum(contagious_camp)==(sum(contagious_sitters)+sum(contagious_wanderers)))
    return contagious_hhl,contagious_hhl_qua,contagious_camp,contagious_sitters,contagious_wanderers,active_camp

def infected_and_sum_by_households(pop_matrix,contagious_hhl,contagious_hhl_qua,contagious_camp,contagious_sitters,contagious_wanderers,active_camp):
    infh = accumarray(pop_matrix[:,0],contagious_hhl)   # All infected in house and at toilets, population 
    infhq = accumarray(pop_matrix[:,0],contagious_hhl_qua) # All infected in house, quarantine 
    infl = accumarray(pop_matrix[:,0],contagious_camp)   # presymptomatic and asymptomatic for food lines
    infls = accumarray(pop_matrix[:,0],contagious_sitters) # All sedentaries for local transmission
    inflw = accumarray(pop_matrix[:,0],contagious_wanderers) # All wanderers for local transmission
    allfl = accumarray(pop_matrix[:,0],active_camp)  # All people in food lines
    return infh,infhq,infl,infls,inflw,allfl

def infected_prob_inhhl(inf_prob_hhl,trans_prob_hhl):
    return 1-(1-trans_prob_hhl)**np.array(inf_prob_hhl) 

def infected_prob_activity(act_sharing_matrix,inf_prob_act,ppl_per_household,occur_per_day,num_contact_per_occur,trans_prob_act,trans_reduction,factor=1):
    #this could be infections at the toilet or the foodline
    proportion_infecteds=(act_sharing_matrix.dot(inf_prob_act))/(act_sharing_matrix.dot(ppl_per_household))
    act_factor=occur_per_day*num_contact_per_occur
    act_coef = np.arange(act_factor+1)
    trans_when_act = 1-factorial(act_factor)*np.sum(((factorial(act_factor-act_coef)*factorial(act_coef))**-1)*
                                (np.transpose(np.array([(1-proportion_infecteds)**(act_factor-i) for i in act_coef])))*
                                (np.transpose(np.array([proportion_infecteds**i for i in act_coef])))*
                                (np.power(1-trans_prob_act*trans_reduction,act_coef)),1)
    return trans_when_act*factor

def infected_prob_movement(pop_matrix,neighbour_inter,infls,inflw,aip,tr):
    lr1_exp_contacts = neighbour_inter[:,:,0].dot(infls)+neighbour_inter[:,:,1].dot(inflw)
    lr2_exp_contacts = neighbour_inter[:,:,1].dot(infls)+neighbour_inter[:,:,2].dot(inflw)    
    # But contacts are roughly Poisson distributed (assuming a large population), so transmission rates are:
    trans_for_lr1 = 1-np.exp(-lr1_exp_contacts*aip*tr)
    trans_for_lr2 = 1-np.exp(-lr2_exp_contacts*aip*tr)    
    # Now, assign the appropriate local transmission rates to each person.
    trans_local_inter = trans_for_lr1[pop_matrix[:,0].astype(int)]*(1-pop_matrix[:,8])+trans_for_lr2[pop_matrix[:,0].astype(int)]*(pop_matrix[:,8])
    return trans_local_inter

def assign_new_infections(pop_matrix,tshared,fshared,num_toilet_visit,num_toilet_contact,num_food_visit,num_food_contact,pct_food_visit,aip,tr,lis,twh):
    ##########################################################
    # IDENTIFY CONTAGIOUS AND ACTVE PEOPLE IN DIFFERENT CONTEXTS.
    # Contagious in the house and at toilets, in population.  
    # At least presymptomatic AND at most severe.
    cpih,cpihq,cpco,cpcos,cpcow,apco=identify_contagious_active(pop_matrix)
    ########################################################## 
    ##########################################################  
    # COMPUTE INFEECTED PEOPLE PER HOUSEHOLD.
    infh,infhq,infl,infls,inflw,allfl=infected_and_sum_by_households(pop_matrix,cpih,cpihq,cpco,cpcos,cpcow,apco)     
    ##########################################################   
    pph = np.bincount(pop_matrix[:,0].astype(int)) #compute people per household
    ##########################################################
    # COMPUTE INFECTION PROBABILITIES FOR EACH PERSON BY HOUSEHOLD.
    # Probability for members of each household to contract from their housemates
    cfh = infected_prob_inhhl(infh,twh)   # In population
    cfhq =infected_prob_inhhl(infhq,twh)   # In quarantine.
    # Compute proportions infecteds at toilets and in food lines.
    trans_at_toil=infected_prob_activity(tshared,infl,allfl,num_toilet_visit,num_toilet_contact,aip,tr)
    # Compute transmission in food lines by household.
    # Assume each person goes to the food line once per day on 75% oft_factor days.
    # Other days someone brings food to them (with no additional contact).
    trans_in_fl=infected_prob_activity(fshared,infl,allfl,num_food_visit,num_food_contact,aip,tr,factor=pct_food_visit)
    # Households in quarantine don't get these exposures, but that is taken care of below
    # because this is applied only to susceptibles in the population with these, we can calculate
    # the probability of all transmissions that calculated at the household level.
    pthl = 1-(1-cfh)*(1-trans_at_toil)*(1-trans_in_fl)
    # Transmissions during movement around the residence must be calculated at the individual level,
    # because they do not depend on what movement radius the individual uses. So...
    # Compute expected contacts with infected individuals for individuals that use small and large movement radii.
    local_trans=infected_prob_movement(pop_matrix,lis,infls,inflw,aip,tr)
    # Finally, compute the full per-person infection probability within households, at toilets and food lines.
    full_inf_prob = 1-((1-pthl[pop_matrix[:,0].astype(int)])*(1-local_trans))
    ##########################################################    
    # ASSIGN NEW INFECTIONS.
    new_inf = full_inf_prob>np.random.uniform(0,1,pop_matrix.shape[0])                # Find new infections by person, population.
    pop_matrix[:,1] += (1-np.sign(pop_matrix[:,1]))*new_inf         # Impose infections, population. Only infect susceptible individuals
    new_infq = cfhq[pop_matrix[:,0].astype(int)]>np.random.uniform(0,1,pop_matrix.shape[0]) # Find new infections by person, quarantine
    pop_matrix[:,1] += (pop_matrix[:,1]==7)*new_infq                         # Impose nfections, quarantine.
    return pop_matrix

def move_hhl_quarantine(pop_matrix,prob_spot_symp):
    spot_symp=np.random.uniform(0,1,pop_matrix.shape[0])<prob_spot_symp
    symp=np.logical_and.reduce((pop_matrix[:,1]>2,pop_matrix[:,1]<6,pop_matrix[:,4]==0,pop_matrix[:,5]>=16))
    spotted_per_day=spot_symp*symp
    symp_house = pop_matrix[spotted_per_day==1,0]
    pop_matrix[np.in1d(pop_matrix[:,0],symp_house),1] += 7
    return pop_matrix
