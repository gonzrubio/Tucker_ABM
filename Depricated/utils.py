import pandas as pd
import numpy as np

def read_age_gender(num_ppl):
    path_to_file = 'age_and_sex.csv'            # Observed data.
    age_and_gender = pd.read_csv(path_to_file)     # Data frame. V1 = age, V2 is sex (1 = male?, 0  = female?).
    age_and_gender = age_and_gender.loc[:, ~age_and_gender.columns.str.contains('^Unnamed')] 
    age_and_gender = age_and_gender.values 
    age_and_gender = age_and_gender[np.random.randint(age_and_gender.shape[0], size=num_ppl)]
    return age_and_gender

