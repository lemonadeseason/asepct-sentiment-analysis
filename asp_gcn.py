import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj

class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, args.num_class)

    def forward(self, inputs,flag=False,mask_pos=-1):
        outputs = self.gcn_model(inputs,flag=flag,mask_pos=mask_pos)
        logits = self.classifier(outputs)
        return logits, outputs

class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix, requires_grad=False)

        self.pos_emb = nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None        # POS emb
        self.post_emb = nn.Embedding(args.post_size, args.post_dim, padding_idx=0) if args.post_dim > 0 else None    # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)

        # gcn layer
        self.gcn = GCN(args, embeddings, args.hidden_dim, args.num_layers)

    def forward(self, inputs,flag=False,mask_pos=-1):
        tok, asp, pos, head, deprel, post, mask, l = inputs           # unpack inputs
        maxlen = max(l.data)

        def inputs_to_tree_reps(head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=self.args.direct, self_loop=self.args.loop).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj

        adj = inputs_to_tree_reps(head.data, tok.data, l.data)
        h = self.gcn(adj, inputs,flag=flag,mask_pos=mask_pos)
        
        # avg pooling asp feature
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                        # aspect words num
        mask = mask.unsqueeze(-1).repeat(1,1,self.args.hidden_dim)    # mask for h
        outputs = (h*mask).sum(dim=1) / asp_wn                        # mask h
        
        return outputs

class GCN(nn.Module):
    def __init__(self, args, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.emb_dim+args.post_dim+args.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings


        # drop out

        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    
    def forward(self, adj, inputs,flag=False,mask_pos=-1):
        tok, asp, pos, head, deprel, post, mask, l = inputs           # unpack inputs
      # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        if flag:
           embs[0,mask_pos] = torch.zeros(embs.size(2))
        #embs = self.in_drop(embs)
        #print(embs[0].sum(-1))
        
        gcn_inputs = embs
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1    # norm
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            #gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
            gcn_inputs = gAxW
        return gcn_inputs



