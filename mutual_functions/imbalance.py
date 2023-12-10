import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.utils import resample
# function to upsample/downsample the data using index
def Sample(y,upsampled = True, minPercentage=0.05):
    #create a dataframe
    df = pd.DataFrame(y,index=range(len(y)),columns=['value'])
    uniq = df['value'].unique()
    # find maximal/miinimal ocuurance
    occurance = np.asarray([len(y[y==i]) for i in uniq])
    ind = np.argwhere(occurance>int(minPercentage*len(y))).flatten()
    uniq = uniq[ind]
    occurance = occurance[ind]

    if upsampled:
        print('upsampling')
        occurance = np.max(occurance)
        replace = True
    else:#downsample
        print('downsampling')
        occurance = np.min(occurance)
        replace = False
    new_df = pd.DataFrame(columns=df.columns)
    for i in uniq:
        #set each of the minority class to a seperate dataframe (iterate over each class)
        df_i = df[df['value'] == i]
        #set other classes to another dataframe
        # other_df = df[df['value'] != i]  
        #upsample the minority class
        df_1_upsampled = resample(df_i,random_state=42,n_samples = occurance,replace=replace)
        #concatenate the upsampled dataframe
        new_df = pd.concat([df_1_upsampled,new_df])
    return new_df.index.tolist()

def imbalance_data(y):

    #create a dataframe
    df = pd.DataFrame(y,index=range(len(y)),columns=['value'])
    uniq = df['value'].unique()
    arr = []
    for i in uniq:
        df_i = df[df['value'] == i]
        arr += df_i.index[range(0,np.random.randint(0,len(df_i)))].tolist()
    return arr