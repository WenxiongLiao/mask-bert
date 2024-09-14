import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
from config.load_param import load_parameter
from build_model.build_ml import set_args
from methods import *

# time.sleep(2500)


parser = argparse.ArgumentParser(
    description='Pytorch Mask Bert Training')

parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dataset', default='snippets', type=str)
parser.add_argument('--task', default=1, type=int)
parser.add_argument('--K_shot', default=5, type=int)
parser.add_argument('--base_batch_size', default=64, type=int)
parser.add_argument('--novel_batch_size', default=8, type=int)
parser.add_argument('--few_shot_tunning_epochs', default=150, type=int)
parser.add_argument('--max_length', default=32, type=int)





args = parser.parse_args()

load_parameter(args)

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
cuda = True if torch.cuda.is_available() else False
args.device = torch.device('cuda:'+str(args.cuda) if cuda else 'cpu')
set_args(args)

LLM_test(args)

# if args.task == 0:
#     #pretrain the model 
#     Pretrain(args)
# elif args.task == 1:
#     #Bert CNN model 
#     BERT_CNN_Model(args)
# elif args.task == 2:
#     #CPFT model 
#     CPFT(args) 
# elif args.task == 3:
#     #FPT model 
#     FPT_BERT_Model(args) 
# elif args.task == 4:
#     # ReInit shot model 
#     Reinit(args) 
# elif args.task == 5:
#     # Mask Bert 
#     Search(args)
#     Mask_BERT_with_ratio(args)
# elif args.task == 6:
#     #NSP BERT 
#     NSP_BERT_Model(args)
# elif args.task == 7:
#     #SNFT BERT 
#     SNFT_Model(args)

# nohup ./scripts/symptoms_5shot.sh >> symptoms_output.log  &
# nohup ./scripts/AG_news_5shot.sh >> AG_news_output.log  &
# nohup ./scripts/nicta_5shot.sh >> nicta_5shot_output.log  &
# nohup python mask_bert.py >> PubMed20k_5shot_Llama2_output.log  &

