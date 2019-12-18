import time  #sleep this Process to better observe
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from lstm import LSTMClassifier
from utils import torch_utils

class LSTMTrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = LSTMClassifier(args, emb_matrix=emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        #self.model.cuda()

        self.optimizer = torch_utils.get_optimizer(args.optim, self.parameters, args.lr)

    # load model_state and args
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    # save model_state and args
    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def update(self, batch):
        # convert to cuda
        #batch = [b.cuda() for b in batch]

        # unpack inputs and label
        #print(batch)
        #print(len(batch))    #batch是有9个tensor的tumple
        inputs = batch[0:8]   #分别代表0-7样本的token，aspect...（gcn中unpack），但是json文件中不够8个？
        label = batch[-1]     #label这个tensor中有32个值，应该代表32个样本的sentiment polarity
        #print(inputs)
        #print(label)
        #time.sleep(600)
        
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, gcn_outputs = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        
        # backward
        loss.backward()
        self.optimizer.step()
        return loss.data, acc

    def predict(self, batch):
        # convert to cuda
       # batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:8]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, gcn_outputs = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        
        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, gcn_outputs.data.cpu().numpy()
