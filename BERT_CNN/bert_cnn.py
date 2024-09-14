import random

import torch
from torch import nn
import numpy as np
from transformers import BertModel, RobertaModel, BertPreTrainedModel,BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
import os
from sklearn.metrics import classification_report,accuracy_score

class BERT_CNN(nn.Module):
    def __init__(self, base_model_save_dir,class_num,device):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(
                        base_model_save_dir)
        self.bert_model.to(device)
        self.device = device 
        self.conv1 = nn.Conv2d(4, 32, [1,768], stride=1)
        self.conv2 = nn.Conv2d(4, 32, [2,768], stride=1)
        self.conv3 = nn.Conv2d(4, 32, [3,768], stride=1)
        self.conv4 = nn.Conv2d(4, 32, [4,768], stride=1)
        self.conv5 = nn.Conv2d(4, 32, [5,768], stride=1)

        self.fc = nn.Linear(5*32, class_num)

    def forward(self, batch_input):
        text = batch_input.pop("text")

        batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
        outputs = self.bert_model(**batch_input,output_hidden_states=True)
        hidden_states = outputs.hidden_states[-4:] 
        hidden_states = torch.concat([hidden_states[0].unsqueeze(0),hidden_states[1].unsqueeze(0),hidden_states[2].unsqueeze(0),hidden_states[3].unsqueeze(0)],dim = 0)
        hidden_states = hidden_states.permute(1, 0, 2, 3)

        conv1_features = self.conv1(hidden_states)
        conv1_features = torch.max(conv1_features,dim = 2).values.squeeze(-1)

        conv2_features = self.conv1(hidden_states)
        conv2_features = torch.max(conv2_features,dim = 2).values.squeeze(-1)


        conv3_features = self.conv1(hidden_states)
        conv3_features = torch.max(conv3_features,dim = 2).values.squeeze(-1)

        conv4_features = self.conv1(hidden_states)
        conv4_features = torch.max(conv4_features,dim = 2).values.squeeze(-1)


        conv5_features = self.conv1(hidden_states)
        conv5_features = torch.max(conv5_features,dim = 2).values.squeeze(-1)


        feature = torch.concat([conv1_features,conv2_features,conv3_features,conv4_features,conv5_features],dim = 1)
        outputs = self.fc(feature)

        return outputs


def test(model,base_class_num,test_loader):

    y_true = []
    y_pred = []
    for batch in test_loader:
        labels = batch["labels"].detach().cpu().numpy() - base_class_num
        with torch.no_grad():
            predictions = model(batch)
        predictions = torch.argmax(predictions, dim=-1).detach().cpu().numpy()
        
        if len(y_true) == 0:
            y_true = labels
            y_pred = predictions
        else:
            y_true = np.concatenate([y_true,labels])
            y_pred = np.concatenate([y_pred,predictions])


    acc = accuracy_score(y_true, y_pred)

    return acc


def train(model,device,model_save_path,epochs,base_class_num,loss_func,optimizer,train_loader,val_loader,test_loader):
    max_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch['labels'] = batch['labels'] - base_class_num
            b_y = batch['labels']
            b_y = b_y.to(device)
            out = model(batch)
            loss = loss_func(out, b_y) 
            
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            optimizer.zero_grad()                           # clear gradients for this training step
            

        model.eval()

        acc = test(model,base_class_num,val_loader)
        if acc > max_acc:
            print('{max_acc}===>>{acc}'.format(max_acc = max_acc , acc = acc))
            max_acc = acc
            torch.save(model, model_save_path)
    
    model = torch.load(model_save_path)
    model.eval()
    # import time
    # begin = time.time()
    test_acc = test(model,base_class_num,test_loader)
    # print('========>>',str(time.time() - begin ))
    print('test acc: ',end='')
    print(test_acc)

        
    return test_acc