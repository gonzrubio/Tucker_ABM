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

