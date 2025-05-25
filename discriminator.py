#!/usr/bin/env python

import torch
from torch import nn

from debias_models.common.classifier import Classifier
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Discriminator(nn.Module):
    def __init__(self,
                 cfg,
                 vocab:          object = None,
                 vocab_size:     int = 0,
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 output_size:    int = 1,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 frequency:       list = None
                 ):

        super(Discriminator, self).__init__()

        self.cfg = cfg
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab = vocab
        # if frequency is None:
        #     frequency = [1]*self.output_size

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax")

       
        if frequency is None:
            self.criterion = nn.NLLLoss( reduction='none') # classification
        else:
            self.criterion = nn.NLLLoss(weight=torch.tensor(frequency), reduction='none') # classification
    def predict(self, logits, **kwargs):
       
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)
    def forward(self, x):
        mask = (x != 1)  # [B,T]
        
        y = self.classifier(x, mask)

        return y
    def reference(self, x, mask, z):
        assert not self.training, "should be in eval mode for prediction"
        y = self.classifier(x, mask, z)
        prob=F.softmax(y, dim=-1)
        return y.argmax(-1), prob
    def get_loss(self, logits, targets, mask=None, **kwargs):

        optional = {}
        loss_vec = self.criterion(logits, targets)  # [B]

        # reweight:
        # weights = torch.tensor([66319, 53252,  3929,  1571])[targets]
        # weights = 1 / weights * 66319
        # loss_vec = loss_vec * weights.to(loss_vec.device)

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["loss"] = ce.item()  # [1]


        return ce, optional