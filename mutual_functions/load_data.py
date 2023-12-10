# from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc,RocCurveDisplay
from sklearn.model_selection import train_test_split
import pickle
from scipy.io.arff import loadarff
import numpy as np

# precision_recall_fscore_support
# import shap
# import seaborn as sns
import matplotlib.pyplot as plt

# import torch
import numpy as np

from usefull_functions import *


# load data from UCR archive (time series classification) amd merge train and test
def UCR_data_loader( data_path,dataset):
    data_path = data_path + dataset + '/'

    if not os.path.exists(os.path.join(data_path + 'X_TRAIN.npy')):#create npy files if not exist
        arffIntoNPY(data_path,dataset,type_ = 'TRAIN')
        arffIntoNPY(data_path,dataset,type_ = 'TEST')
         
    X_train = np.load(os.path.join(data_path + 'X_TRAIN.npy'), allow_pickle = True).astype(float)

    X_test = np.load(os.path.join(data_path + 'X_TEST.npy'), allow_pickle = True).astype(float)

    # if task_type == 'classification':
    y_train = np.load(os.path.join(data_path + 'y_TRAIN.npy'), allow_pickle = True)
    y_test = np.load(os.path.join(data_path + 'y_TEST.npy'), allow_pickle = True)

    x = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))


    return x,y
def arffIntoNPY(data_path,dataset, type_ = 'TRAIN',x_tot = None,y_tot = None,save = True):
    
    for i in range(1000): #we dont know what is the dimension of the data
        try:
            raw_data = loadarff(data_path + f"{dataset}Dimension{i+1}_{type_}.arff")[0]
        except:
            break
        data = [list(r) for r in raw_data]
        
        

        x = np.asarray([r[:-1] for r in data])
        x_tot = np.empty(x.shape) if x_tot is None else x_tot
        x_tot = np.dstack((x_tot,x))
    y_tot = np.asarray([r[-1].decode("utf-8") for r in data])   
    x_tot = x_tot[:,:,1:]
    print(x_tot.shape,y_tot.shape)
    if save:
        np.save(data_path  + f"X_{type_}.npy",x_tot)
        np.save(data_path  + f"y_{type_}.npy",y_tot)
    else:
        return x_tot,y_tot
# x_train,y_train = arffIntoNPY(data_path,prop['dataset'],type_ = 'TRAIN')
# x_test,y_test = arffIntoNPY(data_path,prop['dataset'],type_ = 'TEST')


def reshape_UCR_data(X, y, n_step=10,consecutive=False):
    

    # LEN = X.shape[0] if LEN is None else LEN

    x_ = [];y_=[]
    cols = ['dim_'+ str(i) for i in range(X.shape[-1])]
    for i in range(X.shape[0]):
        Xtmp = pd.DataFrame(X[i],columns = cols)
        if consecutive: #num of series = LEN-n_step+1
            xtmp_ = list(Xtmp.rolling(n_step))
        else: #num of series = LEN//n_step
            xtmp_ = split_df(Xtmp, n_step) 
        xtmp = [item for item in xtmp_ if len(item)==n_step]

        ytmp = [y[i]]*len(xtmp)
        x_.append(xtmp)
        y_.append(ytmp)
    return x_,y_



def train_test_split_stratified(X,y,test_size=0.2,random_state=42,val_size = 0,shuffle = True):
    # split to train and test with stratify       
    stratify = get_first_cell(y)
    
    
    # shuffle==False: split data according to original UCR archive split (for reproducibility) 
    #                   i.e without shuffling it and without stratifying it and without validation dataset
    if not shuffle:
        random_state=None
        stratify = None
        test_size=0.5
        val_size = 0

    train,test = train_test_split(list(zip(X,y)), test_size=test_size, random_state=random_state, stratify=stratify,shuffle = shuffle)#shuffle = False
    X_train,y_train = zip(*train)

    
    if val_size > 0: #divide test into test and validation by 50% each
        val,test = train_test_split(test, test_size=0.5, random_state=random_state, shuffle = True)#stratify=stratify
        X_val,y_val = zip(*val)
        X_val = flatten(X_val)
        y_val = flatten(y_val).astype(np.int64)
    else:
        X_val,y_val = None,None
    X_test,y_test = zip(*test)
    X_train,X_test = flatten(X_train),flatten(X_test)
    y_train = flatten(y_train).astype(np.int64)
    y_test = flatten(y_test).astype(np.int64)
    
    
    return X_train,y_train,X_test,y_test,X_val,y_val
    
def transformed_UCR_data(prop,data_path):
        X,y = UCR_data_loader(data_path,prop['dataset'])
        # split series into more series each with sequence length according to prop['seq_len']: if None, use the whole serie as one sample 
        seq_len = prop['seq_len'] if prop['seq_len'] is not None and prop['seq_len']<X.shape[1] else X.shape[1]
        

        # 1. split series into more series each with length prop['seq_len']
        # 2. data split can be done either as rolling (specific time sample is in multiple series) or consecutive (no diuplicates)
        # 3.transform data structure into list of lists of dataframes; each df is a time series, all list's df originated from the same original time series
        # (used to avoid mixing single time serie's data both in train and test) 
        X,y = reshape_UCR_data(X,y,seq_len,consecutive = prop['consecutive'])
        return X,y
# transform classes (such as 'a','b', 'c) into consecutive numbers (such as 0,1,2)
def classes_to_consecutive_numbers(y):

    # every given class is transformed into a consecutive number
    y_dict = dict()
    for i,class_ in enumerate(np.unique(y)):
        y_dict[class_] = i
    # for list of lists; every given class is transformed into a consecutive number
    y = [[y_dict[cl] for cl in y_] for y_ in y]
    return y_dict,y

def random_sample(x,y,p = None):
    
    if p is None:
        return x,y
    uniq = np.unique(y)
    LEN = (y.shape[0] * p)# wanted number of samples from x,y
    l = int(np.ceil(LEN/len(uniq)))# number of samples from each class
    IND = []
    for i in uniq:
        ind = np.where(y == i)[0]
        np.random.shuffle(ind)# sample from each class
        IND += list(ind[:l]) # add to list of indexes
    return x[IND],y[IND]
   


def loadData(prop,data_path):
    print('Data loading start...')
    # transform data from UCR archive (open source datasets) into shorter time series according to prop['seq_len']
    if prop['UCR']: #DATA FROM UCR ARCHIVE
        X,y = transformed_UCR_data(prop,data_path)
           
    # the actual internal data which is not open source (short time series with a given length)
    else:
        with open (data_path + prop['dataset'], 'rb') as file:    
            X,y = pickle.load(file)
            file.close()
    prop['features'] = list(X[0][0].columns)
    print (f"seq_len (n steps): {len(X[0][0])}, data columns: {prop['features']}")
    prop['class_to_consecutive'],y = classes_to_consecutive_numbers(y)#class_names is dict; dict[class] = i(enumerate)
    # prop['class_names_'],_ = get_class_names(df=pd.read_csv(data_path + prop['dataset'] + '/class.csv', index_col=None,) ,dict_class_names = prop['class_names'].copy())
    prop['nclasses'] = len(np.unique(y))
    

    # split to train and test with stratify, convert classes id numbers into *consecutive* numbers
    # shuffle==False: split data according to original UCR archive split (for reproducibility) 
    #                   i.e without shuffling it and without stratifying it and without validation dataset
    X_train,y_train,X_test,y_test,X_val,y_val = train_test_split_stratified(X,y,test_size=0.2,val_size=prop['val_size'],shuffle = prop['shuffle'])


    

    # ------------------------------------------
    # random sample without creating imbalance in data
    p = prop['size_percentage']
    if p is not None:
        print (f"{p*100}% random sample (without class imbalnce)")
        X_train,y_train = random_sample(X_train,y_train,p)
        X_test,y_test = random_sample(X_test,y_test,p)
        if X_val is not None:
            X_val,y_val = random_sample(X_val,y_val,p)

    # prop['nclasses'] = int(np.max(y_train)) + 1 if prop['task_type'] == 'classification' else None
    # UNITEST: check distribution:
    # get y according to x index
    for i in np.unique(np.hstack((y_train,y_test))): #uniq y values
        print(f'train - {i}: {list(y_train).count(i)}, test - {i}: {list(y_test).count(i)}')
    
    return X_train,y_train,X_test,y_test,X_val,y_val,prop

def convert_into_sktime(x):
    # convert our datastructure into sktime format:
    # input x[i,j,k]; i - number of time series, j - length of time series, k - number of features
    # output xx[i,k] dataframe; each cell is a 1-feature time series of length j
    if x is None: return None
    xx = pd.DataFrame(index = range(x.shape[0]),columns=['dim_' + str(i) for i in range(x.shape[-1])])
    for k in range(x.shape[-1]): # for each feature - k; all time series - i, and all time points - j 
        for i in range(x.shape[0]):
            # each serie i is a 1-feature time series (according to selected k)
            xx.loc[i]['dim_' + str(k)] = pd.Series(x[i,:,k])
    return xx