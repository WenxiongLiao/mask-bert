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
import itertools
import heapq
import nltk
import math
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
import umap


from active_learning.AL import AL_detect,get_feature
from build_model.adamW import AdamW


def set_args(new_args):
    global args
    args = new_args
    init()

def init():
    global tokenizer,device,max_length
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = args.max_length)
    device = args.device
    max_length = args.max_length


def seed_everything(seed_value):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True,max_length = args.max_length)

def collate_function(data):

    text = [d['text'] for d in data]
    labels = [d['labels'] for d in data]
    input_ids = [d['input_ids'] for d in data]
    token_type_ids = [d['token_type_ids'] for d in data]
    attention_mask = [d['attention_mask'] for d in data]

    
    
    max_len = max([len(input_id) for input_id in input_ids])

    input_ids = [input_id if len(input_id) == max_len else np.concatenate([input_id,[102] * (max_len - len(input_id))]) for input_id in input_ids ]
    token_type_ids = [token_type if len(token_type) == max_len else np.concatenate([token_type,[0] * (max_len - len(token_type))]) for token_type in token_type_ids ]
    attention_mask = [mask if len(mask) == max_len else np.concatenate([mask,[0] * (max_len - len(mask))]) for mask in attention_mask ]

    if 'sample_type' in data[0].keys():
        sample_type = [d['sample_type'] for d in data]

        batch = {'input_ids':torch.LongTensor(input_ids),
                'token_type_ids':torch.LongTensor(token_type_ids),
                'attention_mask':torch.LongTensor(attention_mask),
                'sample_type':torch.LongTensor(sample_type),
                'labels':torch.LongTensor(labels),
                'text':text
                }
    else:
        batch = {'input_ids':torch.LongTensor(input_ids),
            'token_type_ids':torch.LongTensor(token_type_ids),
            'attention_mask':torch.LongTensor(attention_mask),
            'labels':torch.LongTensor(labels),
            'text':text
            }

    return batch

def constrative_loss(feature,label,novel_label = None):
    pairwised_consine = torch.cosine_similarity(feature.double().unsqueeze(1),feature.unsqueeze(0),dim = -1)
    pairwised_consine = torch.exp(pairwised_consine)
    pairwised_consine = torch.triu(pairwised_consine,diagonal = 1)
    # print(pairwised_consine)

    denominator = torch.sum(pairwised_consine)

    numerator = pairwised_consine[0,0] 
    max_label = torch.max(label)
    for l in range(max_label + 1):
        l_idx = np.array(list(range(0,len(label))))[label.detach().cpu().numpy() == l]
        if len(l_idx) > 1: 
            pair_idx = list(itertools.combinations(l_idx, 2))
            # print(pair_idx)
            for idx in pair_idx:
                numerator = numerator + pairwised_consine[idx]

    # print(numerator,denominator)
    
    loss = - torch.log( (numerator + 1e-4) / (denominator + 1e-4) ) / len(label)
    
    return loss

def build_optimizer(model,lr,n_epoch,data_loader = None,with_scheduler = False,betas = (0.9,0.999),with_bc=True):
#     optimizer = RMSprop(model.parameters(), lr=lr,momentum = 0, weight_decay = 0.005)
    optimizer = AdamW(model.parameters(), lr=lr,betas=betas,with_bc = with_bc)
#     optimizer = AdamW(model.parameters(), lr=lr)
#     optimizer = AdamW(model.parameters(), lr=lr)
#     optimizer = SGD(model.parameters(), lr=lr, weight_decay = 0.005)

    if with_scheduler:
        assert data_loader!=None, print('params error!')
        # init lr
        num_training_steps = len(data_loader) * n_epoch 

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps= int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        return optimizer,lr_scheduler
    return optimizer,None


def hybrid_train_one_epoch(model,hybrid_train_dataloader,optimizer,top_N,device,lr_scheduler=None,with_pos=False,is_continuous = False,is_constractive = True,with_mask = True,random_mask = False):

    progress_bar = tqdm(range( len(hybrid_train_dataloader)))
#     print('model training...')

    model.train()

    for batch in hybrid_train_dataloader:

        if with_mask:
            mask_batch = get_sentences_word_attributions(batch,model,top_N,with_pos = with_pos,is_continuous = is_continuous,random_mask = random_mask)
        else:
            text = batch.pop("text")
            sample_type = batch.pop("sample_type")
            mask_batch = batch

        mask_batch = {k: v.to(device) for k, v in mask_batch.items()}
        outputs = model(**mask_batch,output_hidden_states=True)



        if is_constractive:
            hidden_states = outputs.hidden_states 
            last_hidden_states = hidden_states[-1]
#             sample_representation = torch.mean(last_hidden_states,dim = 1)
            sample_representation = last_hidden_states[:,0,:]
            cons_loss = constrative_loss(sample_representation,mask_batch['labels'])
            cls_loss = outputs.loss
            loss = cls_loss + cons_loss
        else:
            loss = outputs.loss
        loss.backward()

        optimizer.step()
        if type(lr_scheduler) != type(None):
            lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    return model


def train_one_epoch_without_mask(model,dataloader,optimizer,device,lr_scheduler=None):
    progress_bar = tqdm(range( len(dataloader)))
#     print('model training...')

    model.train()

    for batch in dataloader:
        text = batch.pop("text")
        
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        if type(lr_scheduler) != type(None):
            lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    return model

def test_without_mask(model,dataloader,device,cls = None,return_samples = False):
    acc = 0
    model.eval()
    y_true = []
    y_pred = []
    text_list = []
    for batch in tqdm(dataloader):
        
        text = batch.pop("text")
        text_list.extend(text)
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
        # print(classification_report(y_true, y_pred,digits= 4))

        acc = accuracy_score(y_true, y_pred)

    if return_samples:
        text_samples = pd.DataFrame({'text':text_list,'predictions':y_pred,'labels':y_true})
        return text_samples,acc
    
    return acc




def get_base_neighborhood(model, base_data_path, base_labels,novel_data_path,novel_labels, K_shot,data_collator,random_seed = 2022,n_times = 1,random_select = False,old_text = []):
    '''
    select neighborhood samples from the base dataset
    '''
    torch.cuda.empty_cache()
    base_shot = K_shot * n_times
    novel_presentations = []
    model.eval()
    base_select_presentations = []
    base_select_texts = []
    base_select_labels = []

    if random_select == True:
        for base_label in base_labels:
            raw_dataset = load_dataset("csv", data_files={'base':base_data_path})
            raw_dataset = raw_dataset.filter(lambda example: example['labels'] == base_label)
            raw_dataset = raw_dataset.shuffle(seed=random_seed)
            tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
            base_selected_dataloader = torch.utils.data.DataLoader(dataset = tokenized_dataset['base'], batch_size = base_shot, shuffle=True, collate_fn = data_collator)

            selected_batch = next(iter(base_selected_dataloader))
            for idx in range(random.randint(1,23)):
                selected_batch = next(iter(base_selected_dataloader))

            while True:
                if len(set(selected_batch['text'].copy())) != len(set(selected_batch['text'].copy()).intersection(set(old_text.copy()))):
                    break
                else:
                    selected_batch = next(iter(base_selected_dataloader))

            text = selected_batch.pop("text")
            label = selected_batch["labels"]

            if base_select_texts == []:
                base_select_texts = text
                base_select_labels = label
            else:
                base_select_texts.extend(text)
                base_select_labels = torch.cat((base_select_labels, label), 0)

    else:
    
        #get novel dataset features
        for novel_label in novel_labels:

            raw_dataset = load_dataset("csv", data_files={'novel':novel_data_path})
            raw_dataset = raw_dataset.filter(lambda example: example['labels'] == novel_label)
            raw_dataset = raw_dataset.shuffle(seed=random_seed)
            tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)


            novel_class_dataloader = torch.utils.data.DataLoader(dataset = tokenized_dataset['novel'], batch_size = base_shot, shuffle=False, collate_fn = data_collator)

            for batch in tqdm(novel_class_dataloader):
                text = batch.pop("text")
                label = batch.pop("labels")
                label = label.detach().cpu().numpy()


                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch,output_hidden_states=True)
                    hidden_states = outputs.hidden_states 
                    last_hidden_states = hidden_states[-1]
                    sample_representation = last_hidden_states[:,0,:].detach().cpu().numpy()
                if novel_presentations == []:
                    novel_presentations = sample_representation
                else:
                    novel_presentations = np.concatenate([novel_presentations,sample_representation])
            

        baseset_presentations = []
        baseset_texts = []
        baseset_labels = []
        
        #get base dataset features
        for base_label in base_labels:

            texts = []
            labels = []
            sample_representations = []
            raw_dataset = load_dataset("csv", data_files={'base':base_data_path})
            raw_dataset = raw_dataset.filter(lambda example: example['labels'] == base_label)
            raw_dataset = raw_dataset.shuffle(seed=random_seed)
            tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
            
            if random_select == False:

                base_selected_dataloader = torch.utils.data.DataLoader(dataset = tokenized_dataset['base'], batch_size = 600, shuffle=False, collate_fn = data_collator)
                
                with torch.no_grad():
                    for batch in tqdm(base_selected_dataloader):
                        text = batch.pop("text")
                        label = batch.pop("labels")
                        label = label.detach().cpu().numpy()


                        batch = {k: v.to(device) for k, v in batch.items()}
                        with torch.no_grad():
                            outputs = model(**batch,output_hidden_states=True)
                            hidden_states = outputs.hidden_states 
                            last_hidden_states = hidden_states[-1]
                            sample_representation = last_hidden_states[:,0,:].detach().cpu().numpy()

                        if labels == []:
                            labels = label
                            texts = text
                            sample_representations = sample_representation
                        else:
                            texts.extend(text)
                            labels = np.concatenate([labels,label])
                            sample_representations = np.concatenate([sample_representations,sample_representation])

                    #Filter out outliers
                    class_center = np.mean(sample_representations,axis = 0)
                    distance_center = - np.sum((sample_representations - class_center) ** 2, axis = 1)
                    min_idx =  heapq.nlargest( int(distance_center.shape[0] * 0.95), range(len(distance_center)), distance_center.take)
                    sample_representations = sample_representations[min_idx]
                    texts = np.array(texts)[min_idx].tolist()
                    labels = labels[min_idx]

                    if baseset_presentations == []:
                        baseset_presentations = sample_representations
                        baseset_texts = texts
                        baseset_labels = labels
                    else:
                        baseset_presentations = np.concatenate([baseset_presentations,sample_representations])
                        baseset_texts.extend(texts)
                        baseset_labels = np.concatenate([baseset_labels,labels])

        #select neiborhood samples
        base_center = {}
        for base_label in base_labels:
            base_center[base_label] = np.mean(baseset_presentations[baseset_labels == base_label],axis = 0)
            
        for shot in range(base_shot):
            for base_label in base_labels:
                class_sample_representations = baseset_presentations[baseset_labels == base_label]
                class_sample_texts = np.array(baseset_texts)[baseset_labels == base_label].tolist()
                class_sample_labels = baseset_labels[baseset_labels == base_label]
                

                novel_distances  = np.array([np.sum((rep - novel_presentations)**2) for rep in class_sample_representations])
        
                # if len(base_select_presentations) != 0 and len(base_select_presentations[base_select_labels != base_label]) != 0:
                #     base_distances  = np.array([np.sum((rep - base_select_presentations[base_select_labels != base_label])**2) for rep in class_sample_representations]) / len(base_select_presentations[base_select_labels != base_label]) * len(novel_presentations)

                # else:
                #     base_distances = np.zeros(shape = [class_sample_representations.shape[0]])
                
                distance_center = np.sum((class_sample_representations - base_center[base_label]) ** 2, axis = 1) * len(novel_presentations)
                

                distances = - distance_center + novel_distances #+  0 * base_distances
                max_idx =  np.argmax(distances)
                
                if base_select_presentations == []:
                    base_select_presentations = np.array([class_sample_representations[max_idx]])
                    base_select_texts = [class_sample_texts[max_idx]]
                    base_select_labels = np.append(base_select_labels,class_sample_labels[max_idx])
                else:
                    base_select_presentations = np.append(base_select_presentations,[class_sample_representations[max_idx]],axis = 0)
                    base_select_texts.append(class_sample_texts[max_idx])
                    base_select_labels = np.append(base_select_labels,class_sample_labels[max_idx])
                
                del_idx = np.argwhere(baseset_labels == base_label).flatten()[max_idx]
                baseset_presentations = np.delete(baseset_presentations, del_idx,axis=0)
                baseset_texts = np.delete(baseset_texts, del_idx,axis=0)
                baseset_labels = np.delete(baseset_labels, del_idx,axis=0)

                assert len(baseset_presentations) == len(baseset_texts) and len(baseset_presentations) == len(baseset_labels),print('error')
                
    base_select_labels = torch.LongTensor(base_select_labels)
    sample_type = torch.zeros([len(base_select_labels)])
    neighborhood_data_dict = {"text":base_select_texts,'labels':base_select_labels,'sample_type':sample_type}
    
    torch.cuda.empty_cache()
    
    return neighborhood_data_dict
    
    
    
    

def get_novel_sample(novel_few_shot_path):

    novel_df = pd.read_csv(novel_few_shot_path)
    sample_type = torch.ones([novel_df.shape[0]])
    text = novel_df['text'].values
    labels = torch.LongTensor(novel_df['labels'].values)
    novel_data_dict = {"text":text,'labels':labels,'sample_type':sample_type}

    return novel_data_dict



def add_selected_base_dataloader(model, base_data_path, batch_size, base_labels, K_shot, data_collator,random_seed = 2022):
    '''
    select neighborhood samples from the base dataset
    '''

    base_shot = K_shot * 3
    select_neighborhoods = []

    for base_label in base_labels:
        
        raw_dataset = load_dataset("csv", data_files={'base':base_data_path})
        raw_dataset = raw_dataset.filter(lambda example: example['labels'] == base_label)
        raw_dataset = raw_dataset.shuffle(seed=random_seed)
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
        
        base_selected_dataloader = torch.utils.data.DataLoader(dataset = tokenized_dataset['base'], batch_size = base_shot, shuffle=False, collate_fn = data_collator)

        model.eval()
        minLoss=99999
        with torch.no_grad():
            for batch in tqdm(base_selected_dataloader):
                batch_device = batch
                text = batch_device.pop("text")
                batch_device = {k: v.to(device) for k, v in batch_device.items()}
                
                with torch.no_grad():
                    outputs = model(**batch_device)

                loss = outputs.loss 
                if loss.item()<minLoss and len(text) == base_shot:
                    minLoss=loss.item()
                    selected_batch = batch
                    selected_batch["text"] = text

        if select_neighborhoods == []:
            select_neighborhoods = selected_batch
        else:
            for k,v in selected_batch.items():
                if k == "text":
                    select_neighborhoods[k].extend(v)
                else:
                    if k in ['input_ids', 'token_type_ids', 'attention_mask']:
                        max_len = max(select_neighborhoods[k].shape[1],v.shape[1])
                        len_ = select_neighborhoods[k].shape[1]
                        select_neighborhoods[k] = select_neighborhoods[k] if len_ == max_len else torch.concat([select_neighborhoods[k],torch.zeros(size = [select_neighborhoods[k].shape[0],max_len - len_])],dim = 1)
                        len_ = v.shape[1]
                        v = v if len_ == max_len else torch.concat([v,torch.zeros(size = [v.shape[0],max_len - len_])],dim = 1)
                        
                    select_neighborhoods[k] = torch.cat((select_neighborhoods[k], v), 0)


    for label in base_labels:
        print(label,select_neighborhoods['labels'][select_neighborhoods['labels'] == label].shape)
        
    select_neighborhoods_dataset = Dataset.from_dict(select_neighborhoods)
#     print(base_selected_dataset)
    select_neighborhoods_dataset = DatasetDict({"base_selected":select_neighborhoods_dataset})
    tokenized_datasets = select_neighborhoods_dataset.map(tokenize_function, batched=True)
#     print(tokenized_datasets)
    base_selected_dataloader = DataLoader(
        tokenized_datasets["base_selected"], shuffle=True, batch_size=batch_size, collate_fn = data_collator
    )
    
    return base_selected_dataloader, select_neighborhoods['text']


def get_whole_word_attributions(word_attributions, with_pos = False):
#     input:
#      ('[CLS]', 0.0),
#      ('trap', 0.027725534317037028),
#      ('##ez', -0.17850601948716388),
#      ('##e', 0.024500284336490763),
#      ('software', 0.23354358430515224),
#      ('[SEP]', 0.23354358430515224)

#       output:
#        ('trapeze', -0.042093400277878695, [0,1]),
#       ('software', 0.23354358430515224, [2,3]),
    sentence = []
    whole_word_attributions = []
    word_idxs = []
    i = -1
    while True:
        word_attribution = word_attributions[i]
        whole_word_attribution = []
        word_idx = []
        i = i + 1
        if i >= len(word_attributions):
            break
        whole_word = word_attributions[i][0]
        whole_word_attribution.append(word_attributions[i][1].astype(np.float))
        word_idx.append(i)

        while i + 1 < len(word_attributions) and word_attributions[i + 1][0].startswith('##'):
            i = i + 1
            whole_word = whole_word + word_attributions[i][0][2:]
            whole_word_attribution.append(word_attributions[i][1].astype(np.float))
            word_idx.append(i)

        sentence.append(whole_word)   
        
        whole_word_attributions.append(np.mean(whole_word_attribution))
        word_idxs.append(word_idx)
    
    assert word_idxs[-1][-1] < len(word_attributions),print('word_attributions error')
    
    result = []
    if with_pos:
        pos_tag = np.array(nltk.pos_tag(sentence[1:-1]))
        for whole_word,att,pos,word_idx in zip(sentence[1:-1],whole_word_attributions[1:-1],pos_tag[:,1],word_idxs[1:-1]):  #exclude [CLS] and [SEP]
            if pos in ['JJ','JJR','JJS','NN','NN','NNS','NNP','NNPS','RB','RBR','RBS','UH','VB','VBD','VBG','VBN','VBP','VBZ','WRB']:
                result.append([whole_word,att,word_idx])
    else:
        for whole_word,att,word_idx in zip(sentence[1:-1],whole_word_attributions[1:-1],word_idxs[1:-1]):  #exclude [CLS] and [SEP]
            result.append([whole_word,att,word_idx])
    result = np.array(result,dtype=object)
    
    return result


def get_sentences_word_attributions(batch,bert_model,top_N_ratio,with_abs = False,with_pos = False,is_continuous = False,random_mask = False):
    '''
    choose the top_N_ratio relevant tokens
    with_abs : use the absolute value of the contribution
    with_pos : use part-of-speech tagging to keep only nouns, adjectives, adverbs
    is_continuous: Keep consecutive words , if is_continuous == True the with_pos == False
    '''
    sentences = batch.pop("text")
    sample_type = batch.pop("sample_type")
    attention_mask = batch["attention_mask"]
    input_ids= batch.pop("input_ids")
    labels = batch.pop("labels")


    if is_continuous == True:
        with_pos = False #if is_continuous == True the with_pos == False

    bert_model.eval()
    multiclass_explainer = SequenceClassificationExplainer(model=bert_model, tokenizer=tokenizer)
    # token_mask = []
    token_ids = []
    for i in range(len(sentences)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            input_id = input_ids[i]
            sentence = sentences[i]
            sentence = ' '.join(sentence.split()[0:max_length]) # avoid OOM

            for sep_idx in range(len(input_id) -1,0,-1):
                if input_id[sep_idx] != 102:
                    break

            if sample_type[i] == 0:
                token_id = []
                token_id.append(101)
                if random_mask == True:
                    mask_begin_idx = random.randint(1,sep_idx - int(sep_idx * top_N_ratio))
                    token_id = np.concatenate([token_id,input_id[mask_begin_idx:mask_begin_idx + int(sep_idx * top_N_ratio)]])

                else:
                    word_attributions = np.array(multiclass_explainer(text=sentence))
                    # top_N = math.ceil((len(word_attributions) * top_N_ratio))
                    # print(top_N)
                    # mask = np.zeros(len(word_attributions))
                    
                    if with_abs:
                        word_attributions[:,1] = [abs(eval(attribution[1])) for attribution in word_attributions]
        #             else:
        #                 word_attributions[:,1] = [eval(attribution[1]) for attribution in word_attributions]
        
                    #get whole word  attributions
                    word_attributions = get_whole_word_attributions(word_attributions, with_pos)
                    # print(sentence)
                    # print(word_attributions)
                    top_N = math.ceil((len(word_attributions) * top_N_ratio))

                    if is_continuous == False:
                        tmp = zip(range(len(word_attributions)), word_attributions[:,1].astype(np.float))
                        top_index = heapq.nlargest(top_N, tmp, key=lambda x:x[1])
                        top_index = [top[0] for top in top_index]
                    else:
                        # [whole_word,att,word_idx]
                        max_attr = np.sum(word_attributions[:,1][0:top_N])
                        max_idx = 0
                        for begin_idx in range(1,len(word_attributions) - top_N,1):
                            if np.sum(word_attributions[:,1][begin_idx:begin_idx + top_N]) > max_attr:
                                # print(len(word_attributions[:,1][begin_idx:begin_idx + top_N]))
                                max_attr = np.sum(word_attributions[:,1][begin_idx:begin_idx + top_N])
                                max_idx = begin_idx
                        
                        top_index = list(range(max_idx,max_idx + top_N))
                    
                    # print(multiclass_explainer.predicted_class_name)
                    # multiclass_explainer.visualize()
                    for idx in top_index:
                        # mask[word_attributions[idx][2]] = 1
    #                     print(word_attributions[idx])
                    # mask[0] = 1
                        if len(input_id) >= word_attributions[idx][2][0] + 1:
                            if len(word_attributions[idx][2]) >1:
                                token_id = np.concatenate([token_id,input_id[word_attributions[idx][2][0] : word_attributions[idx][2][-1] + 1]])
                            else:
                                token_id = np.concatenate([token_id,[input_id[word_attributions[idx][2][0]]]])
                
                token_id = np.concatenate([token_id,[102]])

            else:
                token_id = input_id[0:sep_idx + 2]
        token_ids.append(token_id)
        # print(input_id)
        # print(token_id)

    
    max_len = max([len(token_id) for token_id in token_ids])
    max_len = min(max_len,max_length)


    token_type_ids = []
    attention_mask = []
    for token_id in token_ids:
        mask = np.zeros(max_len)
        mask[0:len(token_id)] = 1
        attention_mask.append(mask)

    token_ids = [token_id[0:max_len] for token_id in token_ids]
    input_ids = [token_id if len(token_id) == max_len else np.concatenate([token_id,[102] * (max_len - len(token_id))]) for token_id in token_ids ]
    input_ids =  torch.LongTensor([ids.tolist() for ids in input_ids ])
    input_ids[:,-1] = 102
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids = torch.LongTensor(token_type_ids)
    attention_mask = torch.LongTensor(attention_mask)

    batch = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,'labels':labels}
    bert_model.train()    
    
    return batch


def visualize_Tsne(model,hybrid_train_dataloader,novel_test_dataloader,top_N,device,with_pos=False,is_continuous = False,with_mask = True,test_sample_num = 200):
    #  https://www.deeplearn.me/2137.html
    # step 1.  visualize trainset
    progress_bar = tqdm(range( len(hybrid_train_dataloader)))
#     print('model training...')
    sample_representations = []
    labels = []
    model.eval()
    
    for batch in hybrid_train_dataloader:

        if with_mask:
            mask_batch = get_sentences_word_attributions(batch,model,top_N,with_pos = with_pos,is_continuous = is_continuous)
        else:
            text = batch.pop("text")
            sample_type = batch.pop("sample_type")
            mask_batch = batch

        mask_batch = {k: v.to(device) for k, v in mask_batch.items()}
        with torch.no_grad(): 
            outputs = model(**mask_batch,output_hidden_states=True)


        hidden_states = outputs.hidden_states 
        last_hidden_states = hidden_states[-1]
#             sample_representation = torch.mean(last_hidden_states,dim = 1)
        sample_representation = last_hidden_states[:,0,:].detach().cpu().numpy()
        label = mask_batch["labels"].detach().cpu().numpy()

        if len(labels) == 0:
            labels = label
            sample_representations = sample_representation
        else:
            labels = np.concatenate([labels,label])
            sample_representations = np.concatenate([sample_representations,sample_representation])

        progress_bar.update(1)

    # tsne = TSNE(n_components=2, init='pca', random_state=2022)
    # t0 = time()
    # result = tsne.fit_transform(sample_representations)
    reducer = umap.UMAP(random_state=2022)
    t0 = time()
    result = reducer.fit_transform(sample_representations)

    fig = plot_embedding(result, labels,
                         'UMAP embedding of the training set ')
    plt.show(fig)

    train_sample_num = len(labels)

    # step 2.  visualize testset
    for batch in novel_test_dataloader:

        text = batch.pop("text")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(): 
            outputs = model(**batch,output_hidden_states=True)


        hidden_states = outputs.hidden_states 
        last_hidden_states = hidden_states[-1]
        #             sample_representation = torch.mean(last_hidden_states,dim = 1)
        sample_representation = last_hidden_states[:,0,:].detach().cpu().numpy()
        label = batch["labels"].detach().cpu().numpy()

        if len(labels) == 0:
            labels = label
            sample_representations = sample_representation
        else:
            labels = np.concatenate([labels,label])
            sample_representations = np.concatenate([sample_representations,sample_representation])

        progress_bar.update(1)

    # tsne = TSNE(n_components=2, init='pca', random_state=2022)
    # t0 = time()
    # result = tsne.fit_transform(sample_representations)
    sample_representations = sample_representations[train_sample_num:test_sample_num + train_sample_num]
    labels = labels[train_sample_num:test_sample_num + train_sample_num]
    reducer = umap.UMAP(random_state=2022)
    t0 = time()
    result = reducer.fit_transform(sample_representations)
    print(set(labels))

    fig2 = plot_embedding(result, labels,
                         'UMAP embedding of the testing set')
    plt.show(fig2)

    model.train()
    

def plot_embedding(data, label, title):
    import matplotlib.pyplot as plt

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.axis('off')
    ax.axis('off')

    color1 = plt.cm.Set1(np.array([0,1,2,3,4]) / 5.)
    color2 = plt.cm.Set2(np.array([0,1,2,3,4]) / 5.)
    color3 = plt.cm.Set3(np.array([0,1,2,3,4]) / 5.)
    color = np.concatenate([color1,color2,color3])
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                color = color[label[i]],
                #  color=plt.cm.Set1(label[i] / 4.),
                # color=['black','lightcoral','maroon','red','darkorange','gold','y','yellowgreen','palegreen','limegreen','c','dodgerblue','mediumblue','darkorchid'],
                 fontdict={'weight': 'bold', 'size': 12})
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    return fig



def get_dataset(raw_datasets,random_seed,base_num_labels):
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    novel_train_dataset = tokenized_datasets["few_shot_novel"]
    
    test_valid = tokenized_datasets["novel_test"].train_test_split(seed = random_seed, test_size=0.3)
    novel_test_dataset = test_valid["train"]
    novel_valid_dataset = test_valid["test"]

    labels, input_ids, token_type_ids, attention_mask = novel_train_dataset['labels'],novel_train_dataset['input_ids'],novel_train_dataset['token_type_ids'],novel_train_dataset['attention_mask']
    labels = [l - base_num_labels for l in labels]
    novel_train_dataset = Dataset.from_dict({'labels':labels,'label':labels,'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask':attention_mask})

    labels, input_ids, token_type_ids, attention_mask = novel_test_dataset['labels'],novel_test_dataset['input_ids'],novel_test_dataset['token_type_ids'],novel_test_dataset['attention_mask']
    labels = [l - base_num_labels for l in labels]
    novel_test_dataset = Dataset.from_dict({'labels':labels,'label':labels,'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask':attention_mask})

    labels,  input_ids, token_type_ids, attention_mask = novel_valid_dataset['labels'],novel_valid_dataset['input_ids'],novel_valid_dataset['token_type_ids'],novel_valid_dataset['attention_mask']
    labels = [l - base_num_labels for l in labels]
    novel_valid_dataset = Dataset.from_dict({'labels':labels,'label':labels,'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask':attention_mask})

    
    return novel_train_dataset,novel_valid_dataset,novel_test_dataset


def get_data_loader(raw_datasets,base_batch_size,novel_batch_size,data_collator,random_seed):
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.remove_columns(['text'])

    base_dataloader = DataLoader(
        tokenized_datasets["base"], shuffle=True, batch_size=base_batch_size, collate_fn = data_collator
    )

    novel_dataloader = DataLoader(
        tokenized_datasets["few_shot_novel"], shuffle=True, batch_size=novel_batch_size, collate_fn = data_collator
    )
    
    if 'novel_val' in raw_datasets.keys():
        novel_test_dataloader = DataLoader(
            tokenized_datasets["novel_test"], batch_size=novel_batch_size * 10, collate_fn = data_collator
        )
        novel_eval_dataloader = DataLoader(
            tokenized_datasets["novel_val"], batch_size=novel_batch_size * 10, collate_fn = data_collator
        )

    else:
        test_valid = tokenized_datasets["novel_test"].train_test_split(seed = random_seed, test_size=0.3)

        novel_test_dataloader = DataLoader(
            test_valid["train"], batch_size=novel_batch_size * 10, collate_fn = data_collator
        )
        novel_eval_dataloader = DataLoader(
            test_valid["test"], batch_size=novel_batch_size * 10, collate_fn = data_collator
        )


    return base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader



def get_hybrid_loader(neighborhood_data_dict,novel_data_dict,batch_size,data_collator,with_neighborhood):
    '''
    build hybrid data_loader for base neighborhood data and novel few-shot data
    '''
    if with_neighborhood:
        text = list(neighborhood_data_dict['text'])
        text.extend(novel_data_dict['text'])
        labels = torch.cat(( neighborhood_data_dict['labels'], novel_data_dict['labels']), 0)
        sample_type = torch.cat(( neighborhood_data_dict['sample_type'], novel_data_dict['sample_type']), 0)
    else:
        text = novel_data_dict['text']
        labels = novel_data_dict['labels']
        sample_type = novel_data_dict['sample_type']

    hybrid_train_data_dict = {'text':text,'labels':labels,'sample_type':sample_type}
    # print(hybrid_train_data_dict)
    # print(len(text))
    # print(len(labels))
    # print(len(sample_type))
    hybrid_train_dataset = Dataset.from_dict(hybrid_train_data_dict)
    hybrid_train_dataset = DatasetDict({"train":hybrid_train_dataset})
    tokenized_datasets = hybrid_train_dataset.map(tokenize_function, batched=True)
    hybrid_train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn = data_collator
    )

    return hybrid_train_dataloader


    