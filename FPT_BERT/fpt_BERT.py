import imp
import json 
import re
from pathlib import Path
from tkinter import NONE
from tkinter.messagebox import NO
import unicodedata
import numpy as np
import pandas as pd
import random
import os
from time import time

from datasets import load_dataset,Dataset,DatasetDict
import torch.nn as nn
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD,RMSprop#,AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler
from transformers_interpret import SequenceClassificationExplainer
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
import umap

import math

from active_learning.AL import AL_detect,get_feature
from torch.optim import AdamW


from build_model.build_ml import seed_everything,get_data_loader,collate_function

from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import default_data_collator



def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_function(examples):
    result = tokenizer(examples["text"])

    return result

def pre_train_LM(pre_train_data,model_checkpoint,pre_train_save_path,batch_size,num_train_epochs,device,chunk_size_ = 128):
    
    global tokenizer,chunk_size
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    chunk_size = chunk_size_

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    model.to(device)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)


    pre_train_dataset = load_dataset("csv", data_files={'train':pre_train_data})
    tokenized_datasets = pre_train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "labels"])

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    train_dataloader = DataLoader(
    lm_datasets["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=5e-5)


    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.1,
        num_training_steps=num_training_steps,
    )



    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward() 

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.save_pretrained(pre_train_save_path)
    tokenizer.save_pretrained(pre_train_save_path)



# def test(model,base_class_num,device,test_loader):

#     y_true = []
#     y_pred = []
#     for batch in test_loader:
#         batch.pop('text')
#         batch['labels'] = batch['labels'] - base_class_num
#         labels = batch["labels"].detach().cpu().numpy()
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             predictions = model(**batch)
#         predictions = torch.argmax(predictions.logits, dim=-1).detach().cpu().numpy()
        
#         if len(y_true) == 0:
#             y_true = labels
#             y_pred = predictions
#         else:
#             y_true = np.concatenate([y_true,labels])
#             y_pred = np.concatenate([y_pred,predictions])


#     acc = accuracy_score(y_true, y_pred)

#     return acc

def test(model,dataloader,device,cls = None):
    acc = 0
    model.eval()
    y_true = []
    y_pred = []
    for batch in tqdm(dataloader):
        
        text = batch.pop("text")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        if type(cls) != None:
            tmp = logits[:,cls]
            logits = torch.ones_like(logits) * (-1e5)
            logits[:,cls] = tmp
            
        predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

        if len(y_true) == 0:
            y_true = labels
            y_pred = predictions
        else:
            y_true = np.concatenate([y_true,labels])
            y_pred = np.concatenate([y_pred,predictions])

    if len(y_true) >0:
        acc = accuracy_score(y_true, y_pred)
    
    return acc



def fine_tunning(model,device,model_save_path,epochs,novel_cls,optimizer,train_loader,val_loader,test_loader):
    max_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch.pop('text')
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            optimizer.zero_grad()                           # clear gradients for this training step
            

        model.eval()
        acc = test(model,val_loader,device,cls = novel_cls)
       
        if acc > max_acc:
            print('{max_acc}===>>{acc}'.format(max_acc = max_acc , acc = acc))
            max_acc = acc
            torch.save(model, model_save_path)
    
    model = torch.load(model_save_path)
    model.eval()
    test_acc = test(model,test_loader,device,cls = novel_cls)
    print('test acc: ',end='')
    print(test_acc)

        
    return test_acc