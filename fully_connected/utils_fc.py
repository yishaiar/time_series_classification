import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from fc import * 
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


def tensor_dataset(prop, X_train, y_train,shuffle = True,drop_last = True):
    # input: 
    #   x [np.float32] - shape: (batch_size, seq_len, feature_size)  
    #   y [np.float32] - shape: (batch_size)  value is according to nclasses;1-n

    # output:  tensor dataset

    X_train = torch.as_tensor(X_train).float()
    # X_test = torch.as_tensor(X_test).float()

    
    if prop['task_type'] == 'classification':
        y_train = torch.as_tensor(y_train)
        # y_test = torch.as_tensor(y_test)
    else: #regression
        y_train = torch.as_tensor(y_train).float()
        # y_test = torch.as_tensor(y_test).float()
    generator = DataLoader(dataset = list(zip(X_train, y_train)), 
                                          batch_size=prop['batch'], shuffle = shuffle,
                                          num_workers=prop['num_workers'],drop_last=drop_last)
    return X_train, y_train,generator


def initialize_training(prop):

    best_model = FC_lightning(prop,prop['task_type'],prop['hidden_size1'],prop['hidden_size2'],prop['hidden_size3'],prop['dropout'],
            batch_size = prop['batch'],input_size = prop['input_size'],n_step = prop['n_step'],output_size = prop['nclasses']).to(prop['device'])
    
    # optimizer = getattr(optim, prop['optimizer'])(model.parameters(), lr=prop['lr'])
    # best_optimizer = getattr(optim, prop['optimizer'])(best_model.parameters(), lr=prop['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr = prop['lr'])
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr = prop['lr']) # get new optimiser

    


    print("Optimizer: ", prop['optimizer'])
    return model, optimizer, criterion, best_model, best_optimizer
