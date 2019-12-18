import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj

class LSTMClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.lstm_model = LSTMAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, args.num_class)       

    def forward(self, inputs):
        outputs = self.lstm_model(inputs)
        logits = self.classifier(outputs)
        return logits, outputs

class LSTMAbsaModel(nn.Module):   #aspect-based sentiment analysis
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)   #tok_size是tok_vocab的长度
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix, requires_grad=False)

        self.post_emb = nn.Embedding(args.post_size, args.post_dim, padding_idx=0) if args.post_dim > 0 else None    # position emb
        #embeddings = (self.emb, self.pos_emb, self.post_emb)
        embeddings = (self.emb, self.post_emb)        

        # gcn layer
        self.lstm = LSTM(args, embeddings, args.hidden_dim, args.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l = inputs           # unpack inputs
        maxlen = max(l.data)


        h = self.lstm(inputs)
        #print("h:{}".format(h.size()))
        asp_wn = torch.Tensor(len(h),1)
        #print("asp_wn:{}".format(asp_wn.size()))
        for i in range(len(h)):
           asp_wn[i] = len(h[0])    #不是只管asp的avg pooling，而是所有tokens
        
        outputs = h.sum(dim=1) / asp_wn                        
        return outputs

class LSTM(nn.Module):   #还包含rnn
    def __init__(self, args, embeddings, mem_dim, num_layers):
        super(LSTM, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.emb_dim+args.post_dim     #输入embedding的维度
        self.emb, self.post_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, (args.rnn_hidden), args.rnn_layers, batch_first=True, \
                dropout=args.rnn_dropout, bidirectional=args.bidirect)
        #if args.bidirect:
        #    self.in_dim = args.rnn_hidden * 2   #args.rnn_hidden is num of rnn hidden states
        #else:
        #    self.in_dim = args.rnn_hidden

        # drop out
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l = inputs           # unpack inputs
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)     #cat是拼接张量的操作，dim=0时以行拼接，1时以列拼接
        #由（32,41,300）和（32,41,30）拼接为（330）
        embs = self.in_drop(embs)
        #drop之后还是32,41,330
        # rnn layer
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))
        #经过rnn处理后32,41,100
        #print("gcn_inputs:{}".format(gcn_inputs.size()))
        return gcn_inputs

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0, c0

