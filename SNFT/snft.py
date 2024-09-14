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

    # build text_input
    text_text = [d['text'] for d in data]
    text_labels = [d['labels'] for d in data]
    text_labels = np.array(text_labels) -  base_class_num
    text_input_ids = [d['input_ids'] for d in data]
    text_token_type_ids = [d['token_type_ids'] for d in data]
    text_attention_mask = [d['attention_mask'] for d in data]

    max_len = max([len(input_id) for input_id in text_input_ids])
    text_input_ids = [input_id if len(input_id) == max_len else np.concatenate([input_id,[102] * (max_len - len(input_id))]) for input_id in text_input_ids ]
    text_token_type_ids = [token_type if len(token_type) == max_len else np.concatenate([token_type,[0] * (max_len - len(token_type))]) for token_type in text_token_type_ids ]
    text_attention_mask = [mask if len(mask) == max_len else np.concatenate([mask,[0] * (max_len - len(mask))]) for mask in text_attention_mask ]


    # build label_input
    label_text = [label_sentence[lable] for lable in text_labels]
    label_input_ids = [label_sentence_inputs['input_ids'][lable] for lable in text_labels]
    label_token_type_ids = [label_sentence_inputs['token_type_ids'][lable] for lable in text_labels]
    label_attention_mask = [label_sentence_inputs['attention_mask'][lable] for lable in text_labels]
    
    max_len = max([len(input_id) for input_id in label_input_ids])
    label_input_ids = [input_id if len(input_id) == max_len else np.concatenate([input_id,[102] * (max_len - len(input_id))]) for input_id in label_input_ids ]
    label_token_type_ids = [token_type if len(token_type) == max_len else np.concatenate([token_type,[0] * (max_len - len(token_type))]) for token_type in label_token_type_ids ]
    label_attention_mask = [mask if len(mask) == max_len else np.concatenate([mask,[0] * (max_len - len(mask))]) for mask in label_attention_mask ]


    batch = {'text':
                {'input_ids':torch.LongTensor(text_input_ids),
                'token_type_ids':torch.LongTensor(text_token_type_ids),
                'attention_mask':torch.LongTensor(text_attention_mask),
                'text':text_text,
                'labels':torch.LongTensor(text_labels)
                },
              'label':
                {'input_ids':torch.LongTensor(label_input_ids),
                'token_type_ids':torch.LongTensor(label_token_type_ids),
                'attention_mask':torch.LongTensor(label_attention_mask),
                'text':label_text
                }
            }

    return batch


def batch_loss(text_embeddings,label_embeddings,device):
    """
    batch softmax and CrossEntropyLoss
    Args:
        text_embeddings: float[N, k]
        label_embeddings: float[K, k]
        device
    """
    labels = torch.LongTensor(list(range(0,text_embeddings.shape[0]))).to(device)
    dot_products = torch.mm(text_embeddings, label_embeddings.T)
    
    cross_entropy = nn.CrossEntropyLoss()
    cross_entropy_loss = cross_entropy(dot_products,labels)
    
    return cross_entropy_loss


def test(model,test_loader,device):

    #get label embedding
    label_inputs_tmp = dict(label_sentence_inputs)
    label_input_ids = label_inputs_tmp['input_ids']
    label_token_type_ids = label_inputs_tmp['token_type_ids']
    label_attention_mask = label_inputs_tmp['attention_mask']
    
    max_len = max([len(input_id) for input_id in label_input_ids])
    label_input_ids = [input_id if len(input_id) == max_len else np.concatenate([input_id,[102] * (max_len - len(input_id))]) for input_id in label_input_ids ]
    label_token_type_ids = [token_type if len(token_type) == max_len else np.concatenate([token_type,[0] * (max_len - len(token_type))]) for token_type in label_token_type_ids ]
    label_attention_mask = [mask if len(mask) == max_len else np.concatenate([mask,[0] * (max_len - len(mask))]) for mask in label_attention_mask ]

    label_inputs = {'input_ids':torch.LongTensor(label_input_ids),
                'token_type_ids':torch.LongTensor(label_token_type_ids),
                'attention_mask':torch.LongTensor(label_attention_mask)}
    label_inputs = {k: v.to(device) for k, v in label_inputs.items()}

    with torch.no_grad():
        lable_out = model(**label_inputs).last_hidden_state
        label_embeddings = lable_out[:,0,:]


    # get label embedding and prediction
    y_true = []
    y_pred = []

    for batch in test_loader:
        batch_text = batch['text']
        text = batch_text.pop("text")
        labels = batch_text.pop("labels").detach().cpu().numpy()
        batch_text = {k: v.to(device) for k, v in batch_text.items()}
        with torch.no_grad():
            text_out = model(**batch_text).last_hidden_state
            text_embeddings = text_out[:,0,:]
            dot_products = torch.mm(text_embeddings, label_embeddings.T)
            predictions = torch.argmax(dot_products,dim = 1).detach().cpu().numpy()

        
        if len(y_true) == 0:
            y_true = labels
            y_pred = predictions
        else:
            y_true = np.concatenate([y_true,labels])
            y_pred = np.concatenate([y_pred,predictions])

    acc = accuracy_score(y_true, y_pred)

    return acc


def fine_tune(model,base_labels,novel_labels,id2label,template,device,model_save_path,epochs,optimizer,novel_dataloader,novel_eval_dataloader,novel_test_dataloader):
    from build_model import build_ml
    tokenizer = build_ml.tokenizer
    novel_labels_name = [id2label[novel_label] for novel_label in novel_labels]
    global label_sentence,label_sentence_inputs,base_class_num
    label_sentence = [template.format(label_name = name) for name in novel_labels_name]

    label_sentence_inputs = tokenizer(label_sentence)
    base_class_num = len(base_labels)


    max_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch in novel_dataloader:

            batch_text = batch['text']
            text = batch_text.pop("text")
            batch_text.pop("labels")
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            text_out = model(**batch_text).last_hidden_state
            text_embeddings = text_out[:,0,:]

            batch_label = batch['label']
            text = batch_label.pop("text")
            batch_label = {k: v.to(device) for k, v in batch_label.items()}
            label_out = model(**batch_label).last_hidden_state
            label_embeddings = label_out[:,0,:]


            loss = batch_loss(text_embeddings,label_embeddings,device)

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




