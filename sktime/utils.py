

# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

import sktime.transformations.panel.rocket as rocket
import sktime.classification.deep_learning as deep_learning

# from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier

from sktime.datasets import load_arrow_head  # univariate dataset
from sktime.datasets import load_basic_motions  # multivariate dataset


# from sktime.transformations.panel.rocket import Rocket,MiniRocket
from sklearn.linear_model import RidgeClassifierCV,SGDClassifier



def rocket_classify(X_train_transform, y_train,cls ='ridge'):
    
    if cls == 'ridge':
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    elif cls == 'sgd-logistic':
        classifier = SGDClassifier(loss='log_loss')
    elif cls == 'sgd-svm':
        classifier = SGDClassifier(loss='hinge')
    elif cls == 'sgd-perceptron':
        classifier = SGDClassifier(loss='perceptron')
    else:
        print('no classifier')
        return None
    # classifier = make_pipeline(
    #                         StandardScaler(with_mean=False), 
    #                         classifier
    #                         )

    print(classifier.fit(X_train_transform, y_train))
    return classifier

def predict_cls( X_test_transform, y_test,classifier,scaler):
    if scaler is not None:
        X_test_transform = scaler.transform(X_test_transform)
    y_pred = classifier.predict(X_test_transform)
    print("Test accuracy score of %s is %s" % (classifier.__class__.__name__, accuracy_score(y_test, y_pred)))
    return y_pred

def predict_dl( X_test, y_test,model):
    y_pred = model.predict(X_test)
    print("Test accuracy score of %s(n_epochs=%s) is %s" % (model.__class__.__name__,model.n_epochs, accuracy_score(y_test, y_pred)))
    return y_pred

def rocket_transform(X_train, transform_type = 'Rocket',num_kernels=None):
    
    if 'MultiRocket' in transform_type:# 'MultiRocket','MultiRocketMultivariate'
        num_kernels_  = 6250
    elif 'MiniRocket' in transform_type:# 'MiniRocketMultivariate','MiniRocket'
        num_kernels_ = 10000
    elif transform_type == 'Rocket':
        num_kernels_ = 10000


    else:
        print('not an existing transform')
        return None
    num_kernels = num_kernels_ if num_kernels is None else num_kernels
        
        
    # default num_kernels=10000
    
    if transform_type == 'Rocket':
        tfm = getattr(rocket, transform_type)(num_kernels = num_kernels, random_state=111)
    # elif transform_type == 'minirocket':
    #     tfm = MiniRocket()
    else:
        tfm =  getattr(rocket, transform_type)(num_kernels = num_kernels)
  
    
    print(tfm.fit(X_train))
    return tfm
# # add features to the data
# def duplicateRows(X,columns):

#     x = pd.DataFrame(index = X.index,columns = columns)
#     for col in x.columns:
#         x[col] = X
#     return x
def load_data_sktime(prop,multivariate = True):
    if not multivariate:
        # univariate Series length: 251 Train cases: 36 Test cases: 175 Number of classes: 3
        X_train, y_train = load_arrow_head(split="test", return_X_y=True)
        X_test, y_test = load_arrow_head(split="train", return_X_y=True)
        # X_train = np.asarray([np.array(X_train.loc[i][0]) for i in range(len(X_train))])
        # X_test = np.asarray([np.array(X_test.loc[i][0]) for i in range(len(X_test))])
    else:
        X_train, y_train = load_basic_motions(split="train", return_X_y=True)
        X_test, y_test = load_basic_motions(split="test", return_X_y=True)
        
        # X_train1, y_train = load_arrow_head(split="test", return_X_y=True)
        # X_test1, y_test = load_arrow_head(split="train", return_X_y=True)
        # print( f'data dimensions: train - {X_train1.shape}, test - {X_test1.shape} with series length {len(X_train1.loc[0][0])}')
        # X_train1,X_test1 = duplicateRows(X_train1,X_train.columns),duplicateRows(X_test1,X_train.columns)
        # print( f'data dimensions: train - {X_train1.shape}, test - {X_test1.shape} with series length {len(X_train1.loc[0][0])}')
    
    

    print( f'data dimensions: train - {X_train.shape}, test - {X_test.shape} with series length {len(X_train.loc[0][0])}')
    return X_train, y_train, X_test, y_test,prop

    


def dl_model(model_type = 'InceptionTime',params = {},metrics = 'accuracy',n_epochs = 0,loss = 'categorical_crossentropy'):
    model_types = ['InceptionTime','FCN','CNN','LSTMFCN', 'MLP','TapNet']
    model_types1 = ['MACNN','MCDCNN','ResNet' ]
    if 'n_epochs' in params.keys():
        n_epochs = params['n_epochs']
        del params['n_epochs']
    if 'metrics' in params.keys():
        metrics = params['metrics']
        del params['metrics']
    if 'loss' in params.keys():
        loss = params['loss']
        del params['loss']
        
    
    if model_type not in model_types and model_type not in model_types1: # 'SimpleRNN'
        if model_type == 'SimpleRNN':
            from sktime.classification.deep_learning.rnn import SimpleRNNClassifier
            getattr(getattr(deep_learning,'rnn'), 'SimpleRNNClassifier')
            model = SimpleRNNClassifier(**params,num_epochs = n_epochs)
            return model
        else:
            print('not an existing model')
            return None
    if model_type in model_types1:

        import sktime.classification.deep_learning.resnet as resnet 
        import sktime.classification.deep_learning.macnn as macnn 
        import sktime.classification.deep_learning.mcdcnn as mcdcnn 


    
    model = getattr(getattr(deep_learning,model_type.lower()), model_type + 'Classifier')(**params,n_epochs=n_epochs)
    model.metrics,model.loss = metrics,loss
    print(params,', metrics: ',metrics, ', loss: ',loss)
    return model
# from sktime.classification.deep_learning.macnn import MACNNClassifier