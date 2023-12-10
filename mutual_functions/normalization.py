
import numpy as np


def MinMax_fit(X):#dummy function to be similar to mean_standardize_fit
    return None
def MinMax_transform(X,feature_range=[0, 1]):
    
    # useful for some ML models like the Multi-layer Perceptrons (MLP), 
    # where the back-propagation can be more stable and even faster when input features are min-max scaled
    # This is done feature-wise in an independent way (i.e. each feature and serie is scaled independently of the others)
    # highly influenced by the data's maximum and minimum; if our data contains outliers it is going to be biased.
    

    # transform single serie with shape (n_features, n_timesteps) 
    # into (n_series,n_timesteps, n_features) with n_series = 1


    
    
    if len(X.shape) == 2: 
        X = X[np.newaxis,:,:]
    # swap so normalization would be on the time axis of the time serie and not on the features axis
    min_=np.min(X,axis=1);max_=np.max(X,axis=1)
    X_ = np.moveaxis(X,[1],[0])

    X_std = (X_.copy() - min_) / (max_ - min_)* (feature_range[1] - feature_range[0]) + feature_range[0]
    # return to original shape
    X_std = np.moveaxis(X_std,[0],[1])


    # # unitest:
    # i=8;j=0
    # x0 = X[i,:,j].copy();min_=np.min(x0);max_=np.max(x0)
    # x0 = (x0.copy() - min_) / (max_ - min_)* (feature_range[1] - feature_range[0]) + feature_range[0]
    # argmin_=np.argmin(x0);argmax_=np.argmax(x0); print(argmin_,argmax_)
    # x0 = X_std[i,:,j].copy();argmin_=np.argmin(x0);argmax_=np.argmax(x0); print(argmin_,argmax_)
    return X_std



def mean_standardize_fit(X,print_ = False):
    m1 = np.mean(X, axis = 1) #mean of each serie for each of the features
    mean = np.mean(m1, axis = 0) #overall mean of each feature over all series
    
    s1 = np.std(X, axis = 1)#std of each serie for each of the features
    std = np.mean(s1, axis = 0)#overall std of each feature over all series
    if print_:
        print('mean: ', mean, ' std: ', std," m1 ",m1.shape," s1 ",s1.shape)
    return mean, std



def mean_standardize_transform(X,params):
    mean, std = params
    return (X - mean) / std

def returnSame(X,params):
    return X


def normalizeData(X_train, X_test, X_val, norm = 'MinMax01'):
    # -------------------------------------------
    # normalize data

    if norm =='MinMax01':
        norm_transform = MinMax_transform
        params = [0, 1]#feature_range
    elif norm =='MinMax-11':
        norm_transform = MinMax_transform
        params = [-1, 1]#feature_range

    elif norm =='mean_standardize':
        norm_transform = mean_standardize_transform
        params = mean_standardize_fit(X_train.copy())#returns as params: mean, std 
    
    elif norm =='without_norm':
        norm_transform = returnSame
        params = None    
    
    
    X_train_norm, X_test_norm = norm_transform(X_train.copy(), params), norm_transform(X_test.copy(), params), 
    if X_val is not None:
        X_val_norm = norm_transform(X_val.copy(), params)
    else:  
        X_val_norm = None 

    # --------------------------------------------------------
    # return data


    print(f"preprocessing data using '{norm}' complete...")
    l = f'X_train: {X_train_norm.shape}, X_test: {X_test_norm.shape},'
    l = l + ' X_val: None' if X_val_norm is None else l + ' X_val: {X_val_norm.shape}'
    print (l)

              
    return X_train_norm, X_test_norm, X_val_norm