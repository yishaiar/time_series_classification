import torch
from torch.nn import functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
# ,#cross_entropy,
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy




# build fully connected network for classification/regression with 3 layers
# each layer has  neurons according to parameter 'hidden_size'
# the layer is bult with dropout and batch normalization and relu activation function
class FC(torch.nn.Module):

    def __init__(self,prop,task_type,hidden_size1,hidden_size2,hidden_size3,dropout,activation = torch.nn.ReLU(), 
                 batch_size = 5,input_size = 1,n_step = 1,output_size = 1):
        super(FC, self).__init__()
        self.fc1 = Linear(input_size*n_step, hidden_size1)
        self.fc2 = Linear(hidden_size1, hidden_size2)
        self.fc3 = Linear(hidden_size2, hidden_size3)
        self.fc4 = Linear(hidden_size3, output_size)
        self.dropout = Dropout(dropout)
        self.batch_norm1 = BatchNorm1d(hidden_size1)
        self.batch_norm2 = BatchNorm1d(hidden_size2)
        self.batch_norm3 = BatchNorm1d(hidden_size3)
        self.activation = activation
        self.prop = prop
        self.batch_size = batch_size

    def forward(self, x):
        
        x  = x.float().view(self.batch_size,-1)

        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

# # build the train function
# def train(model, optimizer, criterion, generator, prop):
#     model.train()
#     train_loss = 0
#     correct = 0
#     for batch_idx, (data, target) in enumerate(generator):
#         data, target = data.to(prop['device']), target.to(prop['device'])
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         if prop['task_type'] == 'classification':
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     train_loss /= len(generator.dataset)
#     if prop['task_type'] == 'classification':
#         print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
#             train_loss, correct, len(generator.dataset),
#             100. * correct / len(generator.dataset)))
#     else:
#         print('Train set: Average loss: {:.4f}'.format(
#             train_loss))
#     return train_loss

# build the model using pytorch lightning
class FC_lightning(pl.LightningModule):
    
        def __init__(self,prop,input_size,):
            super(FC_lightning, self).__init__()
            self.save_hyperparameters()
            nclasses,hidden_size1,hidden_size2,hidden_size3,dropout = prop['nclasses'],prop['hidden_size1'],prop['hidden_size2'],prop['hidden_size3'],prop['dropout']
            
            
            self.fc1 = Linear(input_size, hidden_size1)
            self.fc2 = Linear(hidden_size1, hidden_size2)
            self.fc3 = Linear(hidden_size2, hidden_size3)
            self.fc4 = Linear(hidden_size3, nclasses)#num_classes to logits output
            self.dropout = Dropout(dropout)
            self.batch_norm1 = BatchNorm1d(hidden_size1)
            self.batch_norm2 = BatchNorm1d(hidden_size2)
            self.batch_norm3 = BatchNorm1d(hidden_size3)
            activation = torch.nn.ReLU()#todo: add activation to prop
            self.activation = activation
            self.prop = prop
            # self.batch_size = prop['batch_size']
            # self.optimizer = prop['optimizer']
            # self.lr = prop['lr']
            # ------------------------
            # self.criterion = torch.nn.CrossEntropyLoss() if prop['task_type'] == 'classification' else torch.nn.MSELoss() # nn.L1Loss() for MAE
            self.criterion = torch.nn.CrossEntropyLoss()
            self.train_acc = Accuracy(task="multiclass", num_classes=nclasses) 
            self.val_acc = Accuracy(task="multiclass", num_classes=nclasses) 
            self.test_acc = Accuracy(task="multiclass", num_classes=nclasses) 
            
            
        def forward(self, x):
            
            x  = x.float().reshape(x.shape[0],-1)
    
            x = self.fc1(x)
            x = self.batch_norm1(x)
            x = self.activation(x)
            x = self.dropout(x)
    
            x = self.fc2(x)
            x = self.batch_norm2(x)
            x = self.activation(x)
            x = self.dropout(x)
    
            x = self.fc3(x)
            x = self.batch_norm3(x)
            x = self.activation(x)
            x = self.dropout(x)
    
            x = self.fc4(x)
            return x
        # add training_step() and validation_step() for pytorch lightning
        def training_step(self, batch, batch_idx):
            # print(f'train: {batch_idx}, global_step: {self.global_step}')
            x, y = batch
            logits = self(x)#outputs of the last fully connected layer before the softmax layer
            loss = self.criterion(logits, y)#F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.train_acc.update(preds, y) 
            self.log("train_loss", loss, prog_bar=True) 
            return loss 
        def on_train_epoch_end(self): 
            #training accuracy logged after each training epoch and not each batch (such as in test,val)
            # otherwise it would be noisy (they dont change wheights)
            self.log("train_acc", self.train_acc.compute())
        # add  validation_step() for pytorch lightning
        def validation_step(self, batch, batch_idx):
            # print(f'val: {batch_idx}')
            x, y = batch
            logits = self(x)#outputs of the last fully connected layer before the softmax layer
            loss = self.criterion(logits, y)#F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.log('val_loss', loss,prog_bar=True)
            
            self.val_acc.update(preds, y) 
            self.log("val_acc", self.val_acc.compute(), prog_bar=True) 
            
            return loss 
        def test_step(self, batch, batch_idx):
            # print(f'test: {batch_idx}')
            x, y = batch
            logits = self(x)#outputs of the last fully connected layer before the softmax layer
            loss = self.criterion(logits, y)#F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.log('test_loss', loss,prog_bar=True)
            
            self.test_acc.update(preds, y) 
            self.log("test_acc", self.test_acc.compute(), prog_bar=True) 
            
            return loss
        def predict_step(self, batch, batch_idx): #predict by batch (not the whole dataset)
            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            return preds
  
        

        
        
        # add configure_optimizers() for pytorch lightning
        def configure_optimizers(self):
            optimizer =  getattr(optim, self.prop['optimizer'])(self.parameters(), lr=self.prop['lr'])
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.prop['lr'])
            # print(optimizer)
            return optimizer        
     