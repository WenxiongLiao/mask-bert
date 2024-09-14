import torch
from torch import nn
import numpy as np
from transformers import BertModel, BertForNextSentencePrediction
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
import os
from sklearn.metrics import classification_report,accuracy_score
from tqdm.auto import tqdm

from datasets import load_dataset
import pandas as pd
from data_preprocessing.processing import build_few_shot_samples



def collate_function(data):
    # print(second_sentence)
    # print(base_class_num)
    # print(second_sentence_inputs)
    text = [d['text'] for d in data]
    labels = [d['labels'] for d in data]
    input_ids = [d['input_ids'] for d in data]
    token_type_ids = [d['token_type_ids'] for d in data]
    attention_mask = [d['attention_mask'] for d in data]

    new_text = []
    new_labels = []
    new_input_ids = []
    new_token_type_ids = []
    new_attention_mask = []

    for i in range(len(text)):
        text_i = text[i]
        labels_i = labels[i]
        input_ids_i = input_ids[i]
        token_type_ids_i = token_type_ids[i]
        attention_mask_i = attention_mask[i]
        for j in range(len(second_sentence)):
            new_text.append(text_i + second_sentence[j])
            if labels_i == j + base_class_num:
                new_labels.append(0)
            else:
                new_labels.append(1)
            
            new_input_ids.append(np.concatenate((input_ids_i,second_sentence_inputs['input_ids'][j])).tolist())
            new_token_type_ids.append(np.concatenate((token_type_ids_i,second_sentence_inputs['token_type_ids'][j])).tolist())
            new_attention_mask.append(np.concatenate((attention_mask_i,second_sentence_inputs['attention_mask'][j])).tolist())


    max_len = max([len(input_id) for input_id in new_input_ids])

    new_input_ids = [input_id if len(input_id) == max_len else np.concatenate([input_id,[102] * (max_len - len(input_id))]) for input_id in new_input_ids ]
    new_token_type_ids = [token_type if len(token_type) == max_len else np.concatenate([token_type,[0] * (max_len - len(token_type))]) for token_type in new_token_type_ids ]
    new_attention_mask = [mask if len(mask) == max_len else np.concatenate([mask,[0] * (max_len - len(mask))]) for mask in new_attention_mask ]


    batch = {'input_ids':torch.LongTensor(new_input_ids),
            'token_type_ids':torch.LongTensor(new_token_type_ids),
            'attention_mask':torch.LongTensor(new_attention_mask),
            'labels':torch.LongTensor(new_labels),
            'text':new_text
            }

    return batch

def test(model,test_loader,device):
    novel_class_num = len(second_sentence)

    y_true = []
    y_pred = []
    for batch in test_loader:
        binary_labels = batch["labels"].detach().cpu().numpy()
        text = batch.pop("text")
        assert len(binary_labels)%novel_class_num == 0, print('data error!')

        batch = {k: v.to(device) for k, v in batch.items()}
        

        with torch.no_grad():
            out = model(**batch)
            soft_max = torch.softmax(out.logits,dim = 1)

        predictions = []
        labels = []
        for i in range(0,len(text),novel_class_num):
            assert 0 in binary_labels[i:i + novel_class_num] and np.sum(binary_labels[i:i + novel_class_num]) == novel_class_num - 1,print('data error!')
            predictions.append(torch.argmax(soft_max[i:i + novel_class_num,0], dim=-1).detach().cpu().numpy())
            labels.append(np.where(binary_labels[i:i + novel_class_num]==0)[0])

        
        if len(y_true) == 0:
            y_true = labels
            y_pred = predictions
        else:
            y_true = np.concatenate([y_true,labels])
            y_pred = np.concatenate([y_pred,predictions])


    acc = accuracy_score(y_true, y_pred)

    return acc

def pre_train(model,base_labels,novel_labels,id2label,template,device,model_save_path,epochs,optimizer,lr_scheduler,base_dataloader):
    from build_model import build_ml
    tokenizer = build_ml.tokenizer
    base_labels_name = [id2label[base_label] for base_label in base_labels]
    global second_sentence,second_sentence_inputs,base_class_num
    second_sentence = [template.format(label_name = name) for name in base_labels_name]

    second_sentence_inputs = tokenizer(second_sentence)
    # base_class_num = len(base_labels)
    base_class_num = 0
    second_sentence_inputs['input_ids'] = [input_id[1:] for input_id in second_sentence_inputs['input_ids']]
    second_sentence_inputs['token_type_ids'] = [token_type_id[1:] for token_type_id in second_sentence_inputs['token_type_ids']]
    second_sentence_inputs['token_type_ids'] = [np.ones_like(token_type_id).tolist() for token_type_id in second_sentence_inputs['token_type_ids']]
    second_sentence_inputs['attention_mask'] = [attention_mask[1:] for attention_mask in second_sentence_inputs['attention_mask']]
    
    num_update_steps_per_epoch = len(base_dataloader)
    num_training_steps = epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(epochs):
        model.train()
        for batch in base_dataloader:
            text = batch.pop("text")

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            optimizer.zero_grad()                           # clear gradients for this training step
            progress_bar.update(1)
            lr_scheduler.step()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


def fine_tune_train(model,base_labels,novel_labels,id2label,template,device,model_save_path,epochs,optimizer,novel_dataloader,novel_eval_dataloader,novel_test_dataloader):
    from build_model import build_ml
    tokenizer = build_ml.tokenizer
    novel_labels_name = [id2label[novel_label] for novel_label in novel_labels]
    global second_sentence,second_sentence_inputs,base_class_num
    second_sentence = [template.format(label_name = name) for name in novel_labels_name]

    second_sentence_inputs = tokenizer(second_sentence)
    base_class_num = len(base_labels)
    second_sentence_inputs['input_ids'] = [input_id[1:] for input_id in second_sentence_inputs['input_ids']]
    second_sentence_inputs['token_type_ids'] = [token_type_id[1:] for token_type_id in second_sentence_inputs['token_type_ids']]
    second_sentence_inputs['token_type_ids'] = [np.ones_like(token_type_id).tolist() for token_type_id in second_sentence_inputs['token_type_ids']]
    second_sentence_inputs['attention_mask'] = [attention_mask[1:] for attention_mask in second_sentence_inputs['attention_mask']]

    max_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch in novel_dataloader:
            text = batch.pop("text")

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            # print(loss)

            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            optimizer.zero_grad()                           # clear gradients for this training step
            

        model.eval()

        acc = test(model,novel_eval_dataloader,device)
        if acc > max_acc:
            print('epoch:{epoch}:{max_acc}===>>{acc}'.format(epoch = epoch, max_acc = max_acc , acc = acc))
            max_acc = acc
            torch.save(model, model_save_path)
    
    model = torch.load(model_save_path)
    model.eval()
    test_acc = test(model,novel_test_dataloader,device)
    print('test acc: ',end='')
    print(test_acc)
        
    return test_acc

