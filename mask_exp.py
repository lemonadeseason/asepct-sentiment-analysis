#load模型(gcn)，加载数据（example1.json），用asp_gcn里的GCNAbsaModel计算每次的h（包括正常embedding和其余不正常的）
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from vocab import Vocab
from utils import helper
from shutil import copyfile
from draw import draw_curve
from sklearn import metrics
from loader import DataLoader
from asp_gcn_trainer import GCNTrainer
from torch.autograd import Variable
from load_w2v import load_pretrained_embedding
from utils import torch_utils, helper

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='saved_models_gcn/best_model.pt', help='Directory of the model.')

parser.add_argument('--data_dir', type=str, default='dataset/Restaurants')
parser.add_argument('--vocab_dir', type=str, default='dataset/Restaurants')
parser.add_argument('--glove_dir', type=str, default='dataset/glove')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)

parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adamax', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models_gcn', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()



# load vocab 
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_tok.vocab')    # token
post_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_post.vocab')    # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pos.vocab')      # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_dep.vocab')      # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pol.vocab')      # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)



# load data
train_batch = DataLoader(args.data_dir + '/example1.json', 1, args, vocab)



# load model
print("Loading model from {}".format(args.model_dir))
opt = torch_utils.load_config(args.model_dir)
loaded_model = GCNTrainer(opt)
loaded_model.load(args.model_dir)


#进行mask实验
for i, batch in enumerate(train_batch):
   loaded_model.mask_exp(batch)
        





