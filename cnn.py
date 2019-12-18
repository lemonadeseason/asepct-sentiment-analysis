import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj

class CNNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.lstm_model = CNNAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(100, args.num_class)       

    def forward(self, inputs):
        outputs = self.lstm_model(inputs)
        logits = self.classifier(outputs)
        return logits, outputs

class CNNAbsaModel(nn.Module):   #aspect-based sentiment analysis
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)   #tok_size是tok_vocab的长度
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix, requires_grad=False)

        self.post_emb = nn.Embedding(args.post_size, args.post_dim, padding_idx=0) if args.post_dim > 0 else None    # position emb


        self.in_dim = args.emb_dim+args.post_dim     #输入embedding的维度
     

        # cnn layer
        self.conv=nn.Conv2d(1,100,(3,self.in_dim))        #kernel_size=3,kernel_num=100

        

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l = inputs           # unpack inputs
        maxlen = max(l.data)
        
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2) 
        
        inputs = embs
        inputs = inputs.unsqueeze(1)
        inputs = F.relu(self.conv(inputs)).squeeze(3)
        inputs = F.max_pool1d(inputs,inputs.size(2)).squeeze(2)
        
        
                      
        return inputs


