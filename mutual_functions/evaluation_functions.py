# from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc,RocCurveDisplay
from sklearn.model_selection import train_test_split
# import pickle
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# precision_recall_fscore_support
# import shap
import seaborn as sns
import matplotlib.pyplot as plt

# import torch
import numpy as np

from usefull_functions import *



# classification evaluation functions:
def evaluate_(pred, target, nclasses, X = None, avg = 'macro',class_names = None,class_transform = None,save = True,model_name =None):
    # input: input is 1 dim np array of ppred, target of shape (batch_size, 1)
    #     target : y true values  
    #     pred: y_pred values (argmax of the nclasses probability i.e logits)
    # avg: 'macro':    Calculate metrics for each label, and find their unweighted mean. 
    #                     This does not take label imbalance into account.
    if save:
        add = os.getcwd()+'/figures/' 
        folderExists(add)
        add = add + model_name + '_' if model_name is not None else add#figure for specific model (when more than single model in folder)
        
    # transform from type/class into consecutive wanted display class/type
    labels=list(class_names.keys())
    target_names=list(class_names.values())
    target = [class_transform[cl] for cl in target]
    pred = [class_transform[cl] for cl in pred]
    # ----------------------------------


    clf_report = classification_report(target, pred,labels=labels, target_names=target_names, output_dict=True)
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T
    fig = plot_heatmap(clf_report,figsize=(1.5*(3+nclasses), 3), title = 'Classification report')
    if save:
        fig.savefig(add +'classification_report.png')

    # ----------------------------------   
    if nclasses == 2:
        fig = plt.figure(figsize=(3, 8))
        fpr, tpr, _ = roc_curve(target, pred)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='ROC curve').plot()
        if save:
            fig.savefig(add +'roc_curve.png')
    else:
        print('roc curve applicable only for binary classification (this is a multiclass classification)')
    # ----------------------------------   

    cm = get_confusion_matrix(class_names,nclasses,target,pred)
    fig = plot_heatmap(cm,figsize=(2*nclasses, 2*nclasses),title = 'Confusion matrix',xlabel = 'Predicted',ylabel = 'True')
    if save:
        fig.savefig(add +'confusion_matrix.png')
    # ----------------------------------   
        


    acc = accuracy_score(target, pred)
    prec =  precision_score(target, pred, average = avg)
    rec = recall_score(target, pred, average = avg)
    f1 = f1_score(target, pred, average = avg)
    results = pd.DataFrame([ acc, prec, rec, f1],index = ['accuracy','precision','recall','f1'],columns = ['score']).T

    fig = plot_heatmap(results,figsize=(6, 2),title = 'Results')
    if save:
        fig.savefig(add +'results.png')        
    
    # ---------------------------------
    # sklearn.feature_selection.mutual_info_classif
    # https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py
    # return results

def get_confusion_matrix(class_names,nclasses,target,pred):
    labels = list(class_names.values()) if class_names is not None else list(range(0, nclasses))
    target = [class_names[cl] for cl in target]
    pred = [class_names[cl] for cl in pred]
    cm = pd.DataFrame(confusion_matrix(target, pred,labels = labels))
    cm.index = labels
    cm.columns = labels
    return cm


   
def plot_heatmap(df,figsize=(15, 15), title = ' ',xlabel = ' ',ylabel = ' '):
    # remove '-','_' so the heatmap will look better 
    df.columns = [col.replace("-", "_" ).replace("_", "\n" )  for col in df.columns]  
    df.index = [col.replace("-", "_" ).replace("_", "\n" )  for col in df.index]  
    
    fig = plt.figure(figsize=figsize)
    sns.heatmap(df, annot = True)#
    plt.title(title)    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig

# def train_test_split_stratify(Tracks,Classes,Types,test_size=0.2,random_state=42,task_type='classification',Class = 'type',val_size = 0,shuffle = True):
#     # split to train and test with stratify       
#     stratify_ = get_first_cell(Types)
#     Types_dict, stratify,Types = classes_to_consecutive_numbers(stratify_,Types)
    
#     if shuffle:
#         train,test = train_test_split(list(zip(Tracks,Classes,Types,stratify,np.arange(len(stratify)))), 
#                                   test_size=test_size, random_state=random_state, stratify=stratify)#shuffle = False
#     if not shuffle:#test_size=0.5,shuffle = False
#             train,test = train_test_split(list(zip(Tracks,Classes,Types,stratify,np.arange(len(stratify)))), 
#                                   test_size=0.5,shuffle = False)#
#     x_train,trainClass,trainType,trainStratify,trainInd = zip(*train)
#     if val_size > 0:
#         val,test = train_test_split(test, test_size=0.5, random_state=random_state, shuffle = True)#stratify=stratify
#         x_val,valClass,valType,valStratify,valInd = zip(*val)
#     x_test,testClass,testType,testStratify,testInd = zip(*test)
    
    

        
    



#     cols = list(x_train[0][0].columns)
#     print(f'data columns {cols}')
#     x_train = flatten(x_train)
#     x_test = flatten(x_test)
#     if val_size > 0:
#         x_val = flatten(x_val)
#     print ('clssified by: ',Class)
#     if task_type == 'classification' and Class == 'type':    
#         y_train = flatten(trainType).astype(np.int64)
#         y_test = flatten(testType).astype(np.int64)
#         if val_size > 0:
#             y_val = flatten(valType).astype(np.int64)
#     else: #Class == 'class'
#         y_train = flatten(trainClass).astype(np.int64)
#         y_test = flatten(testClass).astype(np.int64)
#         if val_size > 0:
#             y_val = flatten(valClass).astype(np.int64)
#     # else: regression
#     #     y_train = flatten(trainClass).astype(float)
#     #     y_test = flatten(testClass).astype(float)
#     #     if val_size > 0:
#     #         y_val = flatten(valClass).astype(float)
#     if val_size > 0:
#         return x_train,y_train,x_test,y_test,x_val,y_val, cols,Types_dict
#     else:
#         return x_train,y_train,x_test,y_test, cols,Types_dict








# def tensor_dataset(prop, X_train, y_train):
#     # input: 
#     #   x [np.float32] - shape: (batch_size, seq_len, feature_size)  
#     #   y [np.float32] - shape: (batch_size)  value is according to nclasses;1-n

#     # output:  tensor dataset

#     X_train = torch.as_tensor(X_train).float()
#     # X_test = torch.as_tensor(X_test).float()

    
#     if prop['task_type'] == 'classification':
#         y_train = torch.as_tensor(y_train)
#         # y_test = torch.as_tensor(y_test)
#     else: #regression
#         y_train = torch.as_tensor(y_train).float()
#         # y_test = torch.as_tensor(y_test).float()
    
#     return X_train, y_train






def get_class_names(df,class_to_consecutive,type_ = 'class'):
    #class_to_consecutive is dict; - the classes are transformed into consecutive numbers
    # i.e class_to_consecutive[class] = class_consecutive_idx

    dict_class_names={}
    type_into_class={}

    if type_ == 'type':
            cols1 = ['type','type_name']
            cols2 = ['type','type']
    else:
        cols1 = ['type','class_name']
        cols2 = ['type','class']

    # create dict from type/class into consecutive wanted display class/type
    for class_,class_consecutive  in class_to_consecutive.items():
        type_into_class[class_consecutive] = df[df[cols2[0]] == class_][cols2[1]].values[0]

    for class_,class_consecutive  in class_to_consecutive.items():
        dict_class_names[type_into_class[class_consecutive]] = df[df[cols1[0]] == class_][cols1[1]].values[0]

    return dict_class_names,type_into_class


# def get_class_names(df,dict_class_names,cols1 = ['type','type_name'],cols2 = ['class','class_name']):
#     dict_class_names_1={}
#     dict_class_names_2={}

#     for key,value  in dict_class_names.items():
#         dict_class_names_2[value] = df[df[cols2[0]] == key][cols2[1]].values[0]
    
#     for key,value  in dict_class_names.items():
#          dict_class_names_1[value] = df[df[cols1[0]] == key][cols1[1]].values[0]
#     for key,value  in dict_class_names.items():
#          dict_class_names_2[value] = df[df[cols2[0]] == key][cols2[1]].values[0]
#     return dict_class_names_1,dict_class_names_2