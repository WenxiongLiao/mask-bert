from pathlib import Path
import numpy as np
import pandas as pd
import random
import os

from datasets import load_dataset,Dataset,DatasetDict
import torch.nn as nn
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification,AutoConfig,DataCollatorWithPadding,BertForSequenceClassification
from transformers import get_scheduler
from transformers import  BertForNextSentencePrediction
from transformers import pipeline
from transformers_interpret import SequenceClassificationExplainer
from datasets import load_metric

from build_model.build_ml import seed_everything,tokenize_function,collate_function,build_optimizer,train_one_epoch_without_mask,test_without_mask,get_data_loader
from data_preprocessing.processing import build_few_shot_samples, data_prep,AG_dbpedia_preprocessing


def load_raw_datasets(novel_path,novel_labels,K_shot,novel_few_shot_path,random_seed,is_AL, base_path, novel_test_path,novel_val_path = None):
    
    build_few_shot_samples(novel_path,novel_labels,K_shot,novel_few_shot_path,random_seed,is_AL = False)
    
    if novel_val_path != None:
        return load_dataset("csv", data_files={'base':base_path,'few_shot_novel':novel_few_shot_path,'novel_test':novel_test_path,'novel_val':novel_val_path})
    else:
        return load_dataset("csv", data_files={'base':base_path,'few_shot_novel':novel_few_shot_path,'novel_test':novel_test_path})
    

def Pretrain(args):

    args.K_shot = 5
    lr = 2e-5
    #step1: Initialize the model by bert-base-cased
    # # define config
    random_seed = 2022
    seed_everything(random_seed)
    config = AutoConfig.from_pretrained(args.model_checkpoint, label2id=args.label2id, id2label=args.id2label)
    bert_model = BertForSequenceClassification.from_pretrained(
                    args.model_checkpoint, config=config)
    bert_model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = args.max_length)


    #step2: load dataset and data processing  (without AL)
    if args.dataset == 'AG_news' or args.dataset == 'dbpedia14':
        AG_dbpedia_preprocessing(args.train_path,args.base_path,args.base_labels)
        AG_dbpedia_preprocessing(args.train_path,args.novel_path,args.novel_labels)
        AG_dbpedia_preprocessing(args.test_path,args.base_test_path,args.base_labels)
        AG_dbpedia_preprocessing(args.test_path,args.novel_test_path,args.novel_labels)
    # elif args.dataset == 'AD':
    #     pass
    else:
        data_prep(args.train_path,args.base_path,args.base_labels)
        data_prep(args.train_path,args.novel_path,args.novel_labels)
        data_prep(args.test_path,args.base_test_path,args.base_labels)
        data_prep(args.test_path,args.novel_test_path,args.novel_labels)

        if args.dev_path != None:
            data_prep(args.dev_path,args.novel_val_path,args.novel_labels)
    
    if args.novel_val_path != None:
        raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,args.K_shot,args.novel_few_shot_path,random_seed,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
    else:
        raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,args.K_shot,args.novel_few_shot_path,random_seed,args.is_AL, args.base_path, args.novel_test_path)

    base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,args.base_batch_size,args.novel_batch_size,collate_function,random_seed)

    # # # step2: fine-tunning on base dataset
    seed_everything(random_seed)
    optimizer,lr_scheduler = build_optimizer(bert_model,lr,args.base_tunning_epochs,base_dataloader,True)
    for epoch_init in range(args.base_tunning_epochs):
        bert_model = train_one_epoch_without_mask(bert_model,base_dataloader,optimizer,args.device,lr_scheduler)

    # save model
    bert_model.save_pretrained(args.base_model_save_dir)
    tokenizer.save_pretrained(args.base_model_save_dir)


def BERT_CNN_Model(args):
    from BERT_CNN.bert_cnn  import BERT_CNN,train
    novel_batch_size = args.novel_batch_size
    device = args.device
    epochs = args.few_shot_tunning_epochs
    K_shot = args.K_shot
    random_states = args.random_states

    novel_few_shot_path = './data/{}/1/BERT_CNN_{}shot_data.csv'.format(args.dataset,K_shot)
    model_save_path = "./save_models/{}_BERT_CNN_{}shot_model.pth".format(args.dataset,K_shot)

    result_df = pd.DataFrame(columns=['K_shot','result'])
    result = []
    for random_state  in random_states:

        seed_everything(random_state)
        if args.novel_val_path != None:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
        else:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path)            
        
        base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,64,novel_batch_size,collate_function,random_state)


        model = BERT_CNN(base_model_save_dir = args.base_model_save_dir,class_num = len(args.novel_labels),device = device)
        model.to(device)
        optimizer = torch.optim.AdamW(params = model.parameters(),lr = 4e-5)
        loss_func = nn.CrossEntropyLoss()

        
        test_acc = train(model,device,model_save_path,epochs,len(args.base_labels),loss_func,optimizer,novel_dataloader,novel_eval_dataloader,novel_test_dataloader)
        result.append(test_acc)
        
    print(result)
    result_df.loc[0] = [K_shot,result]

    result_path = "./results/{}/{}shot_BERT_CNN.csv".format(args.dataset,K_shot)
    save_dir = os.path.dirname(result_path)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    result_df.to_csv(result_path, index = False)  


def LLM_test(args):
    from LLM.LLM_utils  import load_LLM,get_predict_acc
    novel_batch_size = args.novel_batch_size
    device = args.device
    K_shot = args.K_shot
    random_states = args.random_states
    max_length = args.max_length
    id2label = args.id2label
    novel_labels = args.novel_labels
    model_path = '/biolab/liaowx/pyworkspace/icu_gpt/save_models/llama-2-13b-chat'
    tokenizer_path = None
    ellipsis = ' ...'
    
    # model_path = 'tiiuae/falcon-7b-instruct'
    # tokenizer_path = './save_models/falcon-7b-instruct'
    # ellipsis = '...'
    
    model,tokenizer = load_LLM(model_path,tokenizer_path,device)
    novel_few_shot_path = './data/{}/1/LLM_{}shot_data.csv'.format(args.dataset,K_shot)


    result_df = pd.DataFrame(columns=['K_shot','result'])
    result = []
    for random_state  in random_states:

        seed_everything(random_state)
        if args.novel_val_path != None:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
        else:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path)            

        test_acc = get_predict_acc(model,tokenizer,ellipsis,novel_few_shot_path,args.novel_test_path,max_length,id2label,novel_labels,device)
        result.append(test_acc)
        
    print(result)
    result_df.loc[0] = [K_shot,result]

    result_path = "./results/{}/{}shot_LLM.csv".format(args.dataset,K_shot)
    save_dir = os.path.dirname(result_path)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    result_df.to_csv(result_path, index = False)  
            

def CPFT(args):
    from CPFT_pretrain.util.model.RobertaForPretrain import RobertaPretrain_utterence
    from CPFT_pretrain.util.trainer import Trainer
    from CPFT_pretrain.util.dataset import LoadDataset, load_intent_examples

    pre_train_model_path = "./save_models/{}_CPFT_pre_train_model".format(args.dataset)
    

    max_length = args.max_length
    batch_size = 32  #base dataset 
    log_freq = 100
    num_workers = 1
    lr = 5e-5
    epochs = 8
    random_seed = 1999

    seed_everything(random_seed)
    train_dataset = LoadDataset(data_path = args.base_path,seq_len=max_length, mode='train',model_checkpoint = args.model_checkpoint)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = RobertaPretrain_utterence(model_name="bert",device = args.device, model_checkpoint = args.model_checkpoint)

    print("Creating Trainer")
    trainer = Trainer(task='pretrain', model=model, train_dataloader=train_data_loader, test_dataloader=None,
                      lr=lr, betas=(0.9, 0.999), weight_decay=0.01,epoch = epochs,
                      with_cuda=True, cuda_devices=args.device, log_freq=100,
                      distributed = False, local_rank = 0, accum_iter= 1,
                      seed= random_seed, model_name='bert')

    print("Training Start")
    for epoch in range(epochs):
        
        trainer.train(epoch,pre_train_model_path)

        train_dataset = LoadDataset(data_path = args.base_path,seq_len=max_length, mode='train',model_checkpoint = args.model_checkpoint)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        trainer.change_train_data(train_data_loader)    

    from CPFT_finetune.util.model.Classifier_Contrastive import Classifier_Contrastive
    from CPFT_finetune.util.trainer import Trainer
    from CPFT_finetune.util.dataset.dataset import LoadDataset, load_intent_examples


    K_shot = args.K_shot
    batch_size = args.novel_batch_size  #novel dataset 

    
    epochs = args.few_shot_tunning_epochs
    random_states = args.random_states

    
    result = []
    result_df = pd.DataFrame(columns=['K_shot','result'])
    fine_tune_model_path = "./save_models/{}_CPFT_{}shot_fine_tune_model".format(args.dataset,K_shot)
    args.novel_few_shot_path = './data/{}/2/novel_CPFT_{}shot_data.csv'.format(args.dataset,K_shot)
    result_path = "./results/{}/{}shot_CPFT.csv".format(args.dataset,K_shot)

    for k, random_state in enumerate(random_states):

        seed_everything(random_state)
        if args.novel_val_path != None:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
        else:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path)            
        
        base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,64,batch_size,collate_function,random_state)

        model = Classifier_Contrastive(model_name="bert", num_labels=len(args.novel_labels),base_num_labels = len(args.base_labels), pretrained_model_path=pre_train_model_path)



        print("Creating Trainer")
        trainer = Trainer(task='train', model=model, train_dataloader=novel_dataloader, val_dataloader=novel_eval_dataloader, test_dataloader=novel_test_dataloader,
                            lr=lr, betas=(0.9, 0.999), weight_decay=0.01,epoch = epochs,
                            with_cuda=True, cuda_devices=args.device, log_freq=100,
                            distributed = False, local_rank = 0, accum_iter= 1,
                            seed= random_state, model_name='bert')


        print("Training Start")
        for epoch in range(epochs):
            trainer.train(epoch,fine_tune_model_path,base_num_labels = len(args.base_labels))


    #   model.load_state_dict(torch.load(fine_tune_model_path))
        model.model = BertForSequenceClassification.from_pretrained(fine_tune_model_path)
        model.to(args.device)
        test_acc = trainer.evaluation(1, novel_test_dataloader,base_num_labels = len(args.base_labels))
        result.append(test_acc)
        
    result_df.loc[0] = [K_shot,result]

    save_dir = os.path.dirname(result_path)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    result_df.to_csv(result_path, index = False)
    print(result)
        

def FPT_BERT_Model(args):
    from FPT_BERT.fpt_BERT import pre_train_LM,fine_tunning

    pre_train_save_path = "./save_models/{}_pre_train_model".format(args.dataset)
    
    #further pre-training
    pre_batch_size = 32
    pre_num_train_epochs = 8
    lr = 2e-5
    seed_everything(2022)
    pre_train_LM(args.base_path,args.model_checkpoint,pre_train_save_path,pre_batch_size,pre_num_train_epochs,args.device,chunk_size_ = 128)


    # # define config
    random_seed = 2022
    seed_everything(random_seed)
    config = AutoConfig.from_pretrained(args.model_checkpoint, label2id=args.label2id, id2label=args.id2label)
    bert_model = BertForSequenceClassification.from_pretrained(
                    pre_train_save_path, config=config)
    bert_model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = args.max_length)


    raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,2,args.novel_few_shot_path,random_seed,args.is_AL, args.base_path, args.novel_test_path)            
    base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,args.base_batch_size,args.novel_batch_size,collate_function,random_seed)

    # # step2: fine-tunning on base dataset
    seed_everything(random_seed)
    
    optimizer,lr_scheduler = build_optimizer(bert_model,lr,args.base_tunning_epochs,base_dataloader,True)
    for epoch_init in range(args.base_tunning_epochs):
        bert_model = train_one_epoch_without_mask(bert_model,base_dataloader,optimizer,args.device,lr_scheduler)

    # save model
    bert_model.save_pretrained(pre_train_save_path)
    tokenizer.save_pretrained(pre_train_save_path)

    #step3: fine tunning on novel dataset
    K_shot = args.K_shot
    epochs = args.few_shot_tunning_epochs
    random_states = args.random_states

    
    result_df = pd.DataFrame(columns=['K_shot','result'])
    fine_tunning_model_save_path = "./save_models/{}_fpt_BERT_{}shot_model.pth".format(args.dataset,K_shot)
    args.novel_few_shot_path = './data/{}/3/BERT_{}shot_data.csv'.format(args.dataset,K_shot)
    result_path = "./results/{}/{}shot_fpt_BERT.csv".format(args.dataset,K_shot)
    result = []
    for random_state  in random_states:

        seed_everything(random_state)
        if args.novel_val_path != None:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
        else:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path)  

        base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,64,args.novel_batch_size,collate_function,random_state)

        model = BertForSequenceClassification.from_pretrained(
                            pre_train_save_path  , config=config)
        model.to(args.device)
        optimizer = torch.optim.AdamW(params = model.parameters(),lr = 4e-5)

        test_acc = fine_tunning(model,args.device,fine_tunning_model_save_path,epochs,args.novel_labels,optimizer,novel_dataloader,novel_eval_dataloader,novel_test_dataloader)
        result.append(test_acc)
        
    print(result)
    result_df.loc[0] = [K_shot,result]

    save_dir = os.path.dirname(result_path)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    result_df.to_csv(result_path, index = False)   


def Reinit(args):
    device = args.device 
    max_length = args.max_length
    base_batch_size = args.base_batch_size
    novel_batch_size = args.novel_batch_size
    lr = 4e-5
    few_shot_tunning_epochs = args.few_shot_tunning_epochs
    

    #step1: Initialize the model by bert-base-cased
    # # define config
    config = AutoConfig.from_pretrained(args.model_checkpoint, label2id=args.label2id, id2label=args.id2label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = max_length)
    random_states = args.random_states

    reinit_layers = [4,6,8]
    K_shot = args.K_shot
   
    result_df = pd.DataFrame(columns=['K_shot','reinit_layer','result'])
    args.novel_few_shot_path = './data/{}/5/Reinit_BERT_{}shot_data.csv'.format(args.dataset,K_shot)
    # novel_model_save_dir = args.novel_model_save_dir
    novel_model_save_dir = "./save_models/{}_Reinit_BERT_{}shot_novel_model".format(args.dataset,K_shot)
    result_path = "./results/{}/{}shot_Reinit.csv".format(args.dataset,K_shot)
    for j,reinit_layer in   enumerate(reinit_layers):

        result = []
        for random_state in random_states:
            #step2: load dataset and data processing  (without AL)
            if args.novel_val_path != None:
                raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
            else:
                raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path)                    
            base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,base_batch_size,novel_batch_size,collate_function,random_state)

            seed_everything(random_state)
            max_acc = 0
            bert_model = BertForSequenceClassification.from_pretrained(
                            args.base_model_save_dir, config=config)
            bert_model.to(device)

            #reinit
            BertLayerNorm = nn.LayerNorm
            encoder_temp = getattr(bert_model, 'bert')
            for layer in encoder_temp.encoder.layer[-reinit_layer :]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        # Slightly different from the TF version which uses truncated_normal for initialization
                        # cf https://github.com/pytorch/pytorch/pull/5617
                        module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                    elif isinstance(module, BertLayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)

                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()


            optimizer,lr_scheduler = build_optimizer(bert_model,lr,few_shot_tunning_epochs,data_loader =novel_dataloader, with_scheduler = True)
            #step3 fine-tunning on few-shot novel dataset  (without mask neighborhood samples from the base dataset)
            for few_shot_epoch in range(few_shot_tunning_epochs):
                bert_model.train()

                bert_model = train_one_epoch_without_mask(bert_model,novel_dataloader,optimizer,device,lr_scheduler)
                acc = test_without_mask(bert_model,novel_eval_dataloader,device,args.novel_labels)

                if max_acc < acc:
                    print('{max_acc}===>>{acc}'.format(max_acc = max_acc , acc = acc))
                    max_acc = acc
                    bert_model.save_pretrained(novel_model_save_dir)
                    tokenizer.save_pretrained(novel_model_save_dir)
                else:
                    print(acc)

            bert_model = BertForSequenceClassification.from_pretrained(
                            novel_model_save_dir, config=config)
            bert_model.to(device)
            acc = test_without_mask(bert_model,novel_test_dataloader,device,args.novel_labels)
            print(random_state,acc)
            result.append(acc)
        print(result)
        result_df.loc[j] = [K_shot,reinit_layer,result]

        save_dir = os.path.dirname(result_path)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        result_df.to_csv(result_path, index = False)   


def Search(args):
    from build_model.build_ml import seed_everything,tokenize_function,collate_function,build_optimizer,test_without_mask,add_selected_base_dataloader,get_sentences_word_attributions,get_data_loader,get_novel_sample,get_base_neighborhood,get_hybrid_loader,hybrid_train_one_epoch,train_one_epoch_without_mask

    K_shot = args.K_shot
    device = args.device
    max_length = args.max_length
    base_batch_size = args.base_batch_size
    novel_batch_size = args.novel_batch_size
    with_pos = False
    is_AL = False
    is_continuous = True
    is_constractive = True
    with_mask = True
    with_neighborhood = True
    random_select = False
    random_mask = False
    

    top_N_ratios = args.top_N_ratios

    lr = 4e-5
    few_shot_tunning_epochs = args.few_shot_tunning_epochs

    max_layer_acc = 0
    best_ratio = 0

    #step1: Initialize the model by bert-base-cased
    # # define config
    config = AutoConfig.from_pretrained(args.model_checkpoint, label2id=args.label2id, id2label=args.id2label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = max_length)


    result_df = pd.DataFrame(columns=['top_N_ratio','result'])
    novel_model_save_dir = "./save_models/{}_novel_model{}_search".format(args.dataset, args.K_shot)
    novel_few_shot_path = './data/{}/{}shot_search/novel_few_shot_data_search.csv'.format(args.dataset,args.K_shot)
    result_path = "./results/{}/constractive_{}shot.csv".format(args.dataset, K_shot)

    for i, top_N_ratio in enumerate(top_N_ratios):

            result = []
        
            for k, random_state in enumerate(args.random_states):

                seed_everything(random_state)
                #step2: load dataset and data processing


                build_few_shot_samples(args.novel_path,args.novel_labels,K_shot,novel_few_shot_path,random_state,is_AL = False)
                
                novel_data_dict = get_novel_sample(novel_few_shot_path)
                if args.novel_val_path != None:
                    raw_datasets = load_dataset("csv", data_files={'base':args.base_path,'few_shot_novel':novel_few_shot_path,'novel_test':args.novel_test_path,'novel_val':args.novel_val_path})
                else:
                    raw_datasets = load_dataset("csv", data_files={'base':args.base_path,'few_shot_novel':novel_few_shot_path,'novel_test':args.novel_test_path})
                base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,base_batch_size,novel_batch_size,collate_function,random_state)

                max_acc = 0
                bert_model = BertForSequenceClassification.from_pretrained(
                                args.base_model_save_dir, config=config)
                bert_model.to(device)

                old_text = []
                
                if with_neighborhood == True:
                    neighborhood_data_dict = get_base_neighborhood(bert_model, args.base_path, args.base_labels,novel_few_shot_path,args.novel_labels, K_shot,collate_function,random_state,random_select = random_select,old_text = old_text)
                    hybrid_train_dataloader = get_hybrid_loader(neighborhood_data_dict,novel_data_dict,novel_batch_size,collate_function,with_neighborhood)

                else:
                    hybrid_train_dataloader = get_hybrid_loader(None,novel_data_dict,novel_batch_size,collate_function,with_neighborhood)
                
                optimizer,lr_scheduler = build_optimizer(bert_model,lr,n_epoch = few_shot_tunning_epochs,data_loader = hybrid_train_dataloader,with_scheduler = args.with_scheduler,betas = args.betas,with_bc = args.with_bc )        
                
                #step4 fine-tunning on few-shot novel dataset  (with mask neighborhood samples from the base dataset)
                for few_shot_epoch in range(few_shot_tunning_epochs):
                    bert_model.train()
                    
                    bert_model = hybrid_train_one_epoch(bert_model,hybrid_train_dataloader,optimizer,top_N_ratio,device,lr_scheduler = lr_scheduler,with_pos=with_pos,is_continuous = is_continuous,is_constractive = is_constractive,with_mask = with_mask,random_mask = random_mask)
                    acc = test_without_mask(bert_model,novel_eval_dataloader,device,args.novel_labels)

                    if max_acc < acc:
                        print('{max_acc}===>>{acc}'.format(max_acc = max_acc , acc = acc))
                        max_acc = acc
                        bert_model.save_pretrained(novel_model_save_dir)
                        tokenizer.save_pretrained(novel_model_save_dir)
                        if acc == 1:
                            break
                    else:
                        print(acc)

                bert_model = BertForSequenceClassification.from_pretrained(
                                novel_model_save_dir, config=config)
                bert_model.to(device)
                print('test acc: ',end='')
                acc = test_without_mask(bert_model,novel_test_dataloader,device,args.novel_labels)
                print(acc)
                result.append(acc)
            result_df.loc[i]=[top_N_ratio, result ]
            print([top_N_ratio, result ])
            save_dir = os.path.dirname(result_path)
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            result_df.to_csv(result_path, index = False)   

            # for each mask layer, find the best ratio 

            avg_acc = np.mean(result)
            if avg_acc > max_layer_acc:
                max_layer_acc, best_ratio = avg_acc, top_N_ratio
    print("best ratio is :", best_ratio)
    args.ratio = best_ratio

    result_df.to_csv(result_path, index = False)   

def Mask_BERT_with_ratio(args):
    from build_model.build_ml import seed_everything,tokenize_function,collate_function,build_optimizer,test_without_mask,add_selected_base_dataloader,get_sentences_word_attributions,get_data_loader,get_novel_sample,get_base_neighborhood,get_hybrid_loader,hybrid_train_one_epoch,train_one_epoch_without_mask

    K_shot = args.K_shot
    max_length = args.max_length
    base_batch_size = args.base_batch_size
    novel_batch_size = args.novel_batch_size
    few_shot_tunning_epochs = args.few_shot_tunning_epochs
    with_pos = False
    is_AL = False
    is_continuous = True
    novel_model_save_dir = "./save_models/{}_novel_model{}".format(args.dataset, K_shot)
    novel_few_shot_path = './data/{}/{}shot_search/novel_few_shot_data.csv'.format(args.dataset,args.K_shot)
    result_path = "./results/{}/{}shot_constractive_abalation.csv".format(args.dataset, K_shot)
    device = args.device
    top_N_ratio = args.ratio
    random_states = args.random_states

    params_list = [ 
        {'is_constractive':False,'with_mask':False,'with_neighborhood':False,'random_select':True,'random_mask':True,'lr':4e-5,'betas' : (0.9,0.999),'with_scheduler' : False,'with_bc':True}, # bert with_bc
        # {'is_constractive':True,'with_mask':False,'with_neighborhood':False,'random_select':True,'random_mask':True,'lr':4e-5,'betas' : (0.9,0.999),'with_scheduler' : False,'with_bc':True},  # constractive
        # {'is_constractive':False,'with_mask':False,'with_neighborhood':True,'random_select':False,'random_mask':True,'lr':4e-5,'betas' : (0,0),'with_scheduler' : False,'with_bc':True}, # neighborhood
        # {'is_constractive':True,'with_mask':False,'with_neighborhood':True,'random_select':False,'random_mask':True,'lr':4e-5,'betas' : (0,0),'with_scheduler' : False,'with_bc':True},  # constractive neighborhood
        # {'is_constractive':True,'with_mask':False,'with_neighborhood':True,'random_select':True,'random_mask':True,'lr':4e-5,'betas' : (0,0),'with_scheduler' : False,'with_bc':True},   # constractive random neighborhood 
        
        {'is_constractive':True,'with_mask':True,'with_neighborhood':True,'random_select':False,'random_mask':False,'lr':4e-5,'betas' : (0,0),'with_scheduler' : False,'with_bc':True},  # constractive neighborhood mask
        # {'is_constractive':True,'with_mask':True,'with_neighborhood':True,'random_select':False,'random_mask':True,'lr':4e-5,'betas' : (0,0),'with_scheduler' : False,'with_bc':True},   # constractive neighborhood random mask
   
    ]


    #step1: Initialize the model by bert-base-cased
    # # define config
    config = AutoConfig.from_pretrained(args.model_checkpoint, label2id=args.label2id, id2label=args.id2label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding=True, truncation=True,model_max_length = max_length)


    result_df = pd.DataFrame(columns=['is_constractive','with_mask','with_neighborhood','random_select','random_mask','lr','with_bc','result'])

    for i, params in enumerate(params_list):
        print(params)
        is_constractive = params['is_constractive']
        with_mask = params['with_mask']
        with_neighborhood = params['with_neighborhood']
        random_select = params['random_select']
        random_mask = params['random_mask']
        lr = params['lr']
        betas = params['betas']
        with_scheduler = params['with_scheduler']
        with_bc = params['with_bc']
        
        result = []

        for k, random_state in enumerate(random_states):

            seed_everything(random_state)

            build_few_shot_samples(args.novel_path,args.novel_labels,K_shot,novel_few_shot_path,random_state,is_AL = is_AL)
            novel_data_dict = get_novel_sample(novel_few_shot_path)
            if args.novel_val_path != None:
                raw_datasets = load_dataset("csv", data_files={'base':args.base_path,'few_shot_novel':novel_few_shot_path,'novel_test':args.novel_test_path,'novel_val':args.novel_val_path})
            else:
                raw_datasets = load_dataset("csv", data_files={'base':args.base_path,'few_shot_novel':novel_few_shot_path,'novel_test':args.novel_test_path})
        
            base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,base_batch_size,novel_batch_size,collate_function,random_state)
        
            max_acc = 0
            bert_model = BertForSequenceClassification.from_pretrained(
                            args.base_model_save_dir, config=config)
            bert_model.to(device)

            old_text = []

            if with_neighborhood == True:
                neighborhood_data_dict = get_base_neighborhood(bert_model, args.base_path, args.base_labels,novel_few_shot_path,args.novel_labels, K_shot,collate_function,random_state,random_select = random_select,old_text = old_text)
                hybrid_train_dataloader = get_hybrid_loader(neighborhood_data_dict,novel_data_dict,novel_batch_size,collate_function,with_neighborhood)

            else:
                hybrid_train_dataloader = get_hybrid_loader(None,novel_data_dict,novel_batch_size,collate_function,with_neighborhood)
                    
            optimizer,lr_scheduler = build_optimizer(bert_model,lr,n_epoch = few_shot_tunning_epochs,data_loader = hybrid_train_dataloader,betas = betas,with_scheduler = with_scheduler,with_bc = with_bc)        
            

            #step4 fine-tunning on few-shot novel dataset  (with mask neighborhood samples from the base dataset)
            for few_shot_epoch in range(few_shot_tunning_epochs):
                bert_model.train()

                bert_model = hybrid_train_one_epoch(bert_model,hybrid_train_dataloader,optimizer,top_N_ratio,device,lr_scheduler = lr_scheduler,with_pos=with_pos,is_continuous = is_continuous,is_constractive = is_constractive,with_mask = with_mask,random_mask = random_mask)
                acc = test_without_mask(bert_model,novel_eval_dataloader,device,args.novel_labels)

                if max_acc < acc:
                    print('{max_acc}===>>{acc}'.format(max_acc = max_acc , acc = acc))
                    max_acc = acc
                    bert_model.save_pretrained(novel_model_save_dir)
                    tokenizer.save_pretrained(novel_model_save_dir)
                    if acc == 1:
                        break
                else:
                    print(acc)
                
            bert_model = BertForSequenceClassification.from_pretrained(
                            novel_model_save_dir, config=config)
            bert_model.to(device)
            print('test acc: ',end='')
            acc = test_without_mask(bert_model,novel_test_dataloader,device,args.novel_labels)
            print(acc)
            result.append(acc)

        result_df.loc[i] = [is_constractive,with_mask,with_neighborhood,random_select,random_mask,lr,with_bc,result]
        
        save_dir = os.path.dirname(result_path)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        result_df.to_csv(result_path, index = False)   
            
    result_df.to_csv(result_path, index = False)   








def NSP_BERT_Model(args):
    from NSP_BERT.nsp_bert import fine_tune_train,collate_function
    
    device = args.device
    template = 'It is about {label_name}.'
    
    novel_batch_size = args.novel_batch_size
    epochs = 10
    K_shot = args.K_shot
    random_states = args.random_states


    result_df = pd.DataFrame(columns=['K_shot','result'])
    args.novel_few_shot_path = './data/{}/6/NSP_BERT_{}shot_data.csv'.format(args.dataset,K_shot)
    result_path = "./results/{}/{}shot_NSP_BERT.csv".format(args.dataset,K_shot)
    model_save_path = "./save_models/{}_NSP_BERT_model{}.pth".format(args.dataset,K_shot)

    result = []
    for random_state  in random_states:
        
        seed_everything(random_state)
        if args.novel_val_path != None:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
        else:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path)
                                
        base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,64,novel_batch_size,collate_function,random_state)

        model = BertForNextSentencePrediction.from_pretrained(args.model_checkpoint)
        model.to(device)
        optimizer = torch.optim.AdamW(params = model.parameters(),lr = 2e-5)

        
        test_acc = fine_tune_train(model,args.base_labels,args.novel_labels,args.id2label,template,device,model_save_path,epochs,optimizer,novel_dataloader,novel_eval_dataloader,novel_test_dataloader)
        result.append(test_acc)
        
    print(result)
    result_df.loc[0] = [K_shot,result]
    save_dir = os.path.dirname(result_path)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    result_df.to_csv(result_path, index = False)   
        

def SNFT_Model(args):
    from build_model.build_ml import seed_everything,get_data_loader
    from build_model import build_ml
    from SNFT.snft import fine_tune,collate_function
    from transformers import  BertForSequenceClassification,BertModel
    
    
    device = args.device
    
    base_labels = args.base_labels
    novel_labels = args.novel_labels
    id2label = args.id2label
    template = 'It is about {label_name}.'
    base_class_num = len(base_labels)


    # few-shot learning
    novel_batch_size = args.novel_batch_size
    # epochs = args.few_shot_tunning_epochs
    epochs = 50
    K_shot = args.K_shot
    random_states = args.random_states
    result_df = pd.DataFrame(columns=['K_shot','result'])

    args.novel_few_shot_path = './data/{}/7/SNFT_BERT_{}shot_data.csv'.format(args.dataset,K_shot)
    result_path = "./results/{}/{}shot_SNFT_BERT.csv".format(args.dataset,K_shot)
    model_save_path = "./save_models/{}shot_SNFT_model.pth".format(args.dataset,K_shot)

    result = []
    for random_state  in random_states:

        seed_everything(random_state)
        if args.novel_val_path != None:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path,args.novel_val_path)
        else:
            raw_datasets = load_raw_datasets(args.novel_path,args.novel_labels,K_shot,args.novel_few_shot_path,random_state,args.is_AL, args.base_path, args.novel_test_path)

        base_dataloader,novel_dataloader,novel_test_dataloader,novel_eval_dataloader = get_data_loader(raw_datasets,64,novel_batch_size,collate_function,random_state)


        model = BertModel.from_pretrained(args.model_checkpoint)
        model.to(device)
        optimizer = torch.optim.AdamW(params = model.parameters(),lr = 4e-5)

        
        test_acc = fine_tune(model,base_labels,novel_labels,id2label,template,device,model_save_path,epochs,optimizer,novel_dataloader,novel_eval_dataloader,novel_test_dataloader)
        result.append(test_acc)
        
    print(result)
    result_df.loc[0] = [K_shot,result]
    save_dir = os.path.dirname(result_path)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    result_df.to_csv(result_path, index = False)   
        