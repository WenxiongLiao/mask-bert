import argparse
import torch
from config.load_param import load_parameter
from build_model.build_ml import set_args
from methods import *


parser = argparse.ArgumentParser(
    description='Pytorch Mask Bert Training')

parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dataset', default='nicta', type=str)
parser.add_argument('--K_shot', default=5, type=int)
parser.add_argument('--base_batch_size', default=64, type=int)
parser.add_argument('--novel_batch_size', default=5, type=int)
parser.add_argument('--few_shot_tunning_epochs', default=150, type=int)
parser.add_argument('--max_length', default=55, type=int)
parser.add_argument('--ratio', default=0.4, type=float)


args = parser.parse_args()

load_parameter(args)

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
cuda = True if torch.cuda.is_available() else False
args.device = torch.device('cuda:'+str(args.cuda) if cuda else 'cpu')
set_args(args)

Pretrain(args) #fine-tuning on base dataset
Mask_BERT_with_ratio(args)


