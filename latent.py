#!/usr/bin/env python
import pdb

import torch
from torch import nn
import numpy as np

from typing import Union
from debias_models.common.util import get_z_stats
from debias_models.common.classifier import Classifier

from debias_models.common.latent import EPS, DependentLatentModel
from torch.nn.functional import softplus, sigmoid, tanh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = ["LatentRationaleModel"]


class LatentRationaleModel(nn.Module):
    """
    Latent Rationale
    Categorical output version (for SST)

    Consists of:

    p(y | x, z)     observation model / classifier
    p(z | x)        latent model

    """
    def __init__(self,
                 cfg,
                 vocab:          object = None,
                 vocab_size:     int = 0,
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 output_size:    int = 1,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 dependent_z:    bool = False,
                 z_rnn_size:     int = 30,
                 selection:      float = 1.0,
                 lasso:          float = 0.0,
                 lambda_init:    float = 1e-4,
                 lambda_min:     float = 1e-4,
                 lagrange_lr:    float = 0.01,
                 lagrange_alpha: float = 0.99,
                 strategy:       int = 0,
                 frequency:      Union[list, None] = None
                 ):

        super(LatentRationaleModel, self).__init__()

        self.cfg = cfg
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab = vocab
        self.selection = selection
        self.lasso = lasso

        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.lambda_init = lambda_init
        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax")

        if self.dependent_z:
            self.latent_model = DependentLatentModel(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer, strategy=strategy)
        else:
            raise NotImplementedError
            # self.latent_model = IndependentLatentModel(
            #     embed=embed, hidden_size=hidden_size,
            #     dropout=dropout, layer=layer)


        if frequency is None:
            # frequency = [1.0/output_size]*output_size
            self.criterion = nn.NLLLoss(reduction='none') # classification
        else:
            self.criterion = nn.NLLLoss(reduction='none', weight=1.0/torch.tensor(frequency)) # classification

        # lagrange
        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.lambda_min = self.lambda_init / 10
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        # self.register_buffer('lambda_min', torch.full((1,), lambda_min))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average
        self.strategy = strategy

    @property
    def z(self):
        return self.latent_model.z

    @property
    def z_layer(self):
        return self.latent_model.z_layer

    @property
    def z_dists(self):
        return self.latent_model.z_dists

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)

    def forward(self, x):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)  # [B,T]
        z = self.latent_model(x, mask)
        y = self.classifier(x, mask, z)

        return y, z, z

    def get_loss(self, logits, targets, mask=None, **kwargs):

        optional = {}
        selection = self.selection
        lasso = self.lasso

        loss_vec = self.criterion(logits, targets)  # [B]


        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item()  # [1]


        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # z = self.generator.z.squeeze()
        z_dists = self.latent_model.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf_0_5 = z_dists[0].cdf(0.5)
            raise NotImplementedError
        else:
            if self.strategy == 2:
                cdf_0_5 = []
                for t in range(len(z_dists)):
                    cdf_t = z_dists[t].cdf(0.5)
                    cdf_0_5.append(cdf_t)
                cdf_0_5 = torch.stack(cdf_0_5, dim=1)
                p0 = cdf_0_5
            else:
                raise NotImplementedError

        p0 = p0.squeeze(-1)
        p0 = torch.where(mask, p0, p0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        pdf_nonzero = 1. - p0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        # c0_hat = l0 - selection
        if self.cfg['abs']:
            c0_hat = torch.abs(l0 - selection)
        else:
            c0_hat = l0 - selection

        # moving average of the constraint
        self.c0_ma = self.lagrange_alpha * self.c0_ma + \
            (1 - self.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = max(self.lambda0 * torch.exp(self.lagrange_lr * c0.detach()), self.lambda_min)

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        loss = ce + self.lambda0.detach() * c0

        if lasso > 0.:
            raise NotImplementedError
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            zt_zero = p0[:, :-1]
            ztp1_nonzero = pdf_nonzero[:, 1:]

            # cost z_t = non-zero, z_{t+1} = zero
            zt_nonzero = pdf_nonzero[:, :-1]
            ztp1_zero = p0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            lasso_cost = lasso_cost * mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange coherence dissatisfaction (batch average)
            target1 = lasso

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = (lasso_cost - target1)

            # update moving average
            self.c1_ma = self.lagrange_alpha * self.c1_ma + \
                (1 - self.lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(
                self.lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["target1"] = lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.lambda1.item()
                optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

            loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, mask, strategy=self.strategy)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional