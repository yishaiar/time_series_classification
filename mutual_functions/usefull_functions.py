import numpy as np
import torch
import pandas as pd
import os
import math
# requirement.txt export:
# python3 -m  pipreqs.pipreqs 

class namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def unzip(arr):
    return zip(*arr)


# list of lists to single list
def flatten(l):         
    data =  [item for sublist in l for item in sublist]
    data  = np.asarray(data).astype(float)
    return data
# np array to tensor
def to_tensor(arr):
    data =  torch.from_numpy(arr).float()
    return data.view(data.shape[0],-1)

# get first cell of every sublist into list and convert to np array
def get_first_cell(arr):
    return np.asarray([item[0] for item in arr])    

def split_df(df, n):
    arr=[]
    for i in range(len(df)//n):
        arr.append(df[i*n:(i+1)*n])
    return arr




# loading user-specified hyperparameters
def get_user_specified_hyperparameters(args):

    prop = {}
    prop['batch'], prop['lr'], prop['nlayers'], prop['emb_size'], prop['nhead'], prop['task_rate'], prop['masking_ratio'], prop['task_type'] = \
        args.batch, args.lr, args.nlayers, args.emb_size, args.nhead, args.task_rate, args.masking_ratio, args.task_type
    return prop



# loading fixed hyperparameters
def get_fixed_hyperparameters(prop, args):
    
    prop['lamb'], prop['epochs'], prop['ratio_highest_attention'], prop['avg'] = args.lamb, args.epochs, args.ratio_highest_attention, args.avg
    prop['dropout'], prop['nhid'], prop['nhid_task'], prop['nhid_tar'], prop['dataset'] = args.dropout, args.nhid, args.nhid_task, args.nhid_tar, args.dataset
    return prop


def get_prop(args):
    
    # loading optimized hyperparameters
    # prop = get_optimized_hyperparameters(args.dataset)

    # loading user-specified hyperparameters
    prop = get_user_specified_hyperparameters(args)
    
    # loading fixed hyperparameters
    prop = get_fixed_hyperparameters(prop, args)
    return prop

    





printGpuMemory = lambda device: print('GPU memory allocated: {:.3f} GB'.format(torch.cuda.memory_allocated(device)/1e9) 
                                      if torch.cuda.is_available() else 'GPU is not available')
# how to use it: printGpuMemory(prop['device'])

# clear large objects from memory
import gc

    


# get the number of parameters of a model (neurons)
def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# how to use it: get_n_params(model)

# find large tensor objects from memory and their size, count number of elements
def find_large_tensors(size_in_mb=100):
    for i, obj in enumerate(gc.get_objects()):
        counter=0
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)) and obj.nelement()*obj.element_size()/1e6>size_in_mb:
            counter+=1
            print(i, type(obj), obj.size(), obj.device, obj.dtype, obj.shape, obj.numel())
            print(obj.nelement()*obj.element_size()/1e6)#in MB
            # break
            # print obj variable name
               
    print(counter)

# clear large tensors from memory
def find_large_tensors(size_in_mb=65,clear=False):
    import gc
    torch.cuda.empty_cache()
    counter=0
    for i, obj in enumerate(gc.get_objects()):
        
        try: 
            # (hasattr(obj, 'data') and torch.is_tensor(obj.data))
            if torch.is_tensor(obj)  and obj.nelement()*obj.element_size()/1e6>size_in_mb:
                # o = obj.cpu()
                
               
                counter+=1
                print(i, type(obj), obj.size(), obj.numel(), obj.nelement()*obj.element_size()/1e6,  ' mb')#in MB
                # a = obj
                # obj.device,obj.dtype
                if clear:
                   
                    del obj
                    
                    gc.collect()
                    
                    
        except:
            pass
    print(counter) 
    # return a       
        
            
            
# find_large_tensors(size_in_mb=60,clear=False)
# a = torch.zeros(10000,device=torch.device('cpu'))



    
# find  object referenced by gc.get_objects and delete it
# def find_large_tensors(size_in_mb=1,clear=False):
    
    
    
    
# v= [[key,value] for key,value in locals().items() if  type(value)==type(object) ]#'__' not in key and
# type(value) ==
# q = [[key,value] for key,value in locals()['globals']().items() if  type(value)==type(object) ]


def folderExists(path):
  if not os.path.exists(path):
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")