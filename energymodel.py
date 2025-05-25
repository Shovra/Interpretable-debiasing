#!/usr/bin/env python

import math
import pdb

import torch
from torch import nn

from debias_models.common.util import get_z_stats

from torch.nn import CrossEntropyLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = ["LatentRationaleModel"]


class EnergyModel(nn.Module):
    """
    Latent Rationale
    Categorical output version (for SST)

    Consists of:

    p(y | x, z)     observation model / classifier
    p(z | x)        latent model

    """
    def __init__(self,
                 cfg,
                 outcome_model,
                 bias_model,
                 sparsity=0.0,
                 coherence=0.0,
                 
                 ):

        super(EnergyModel, self).__init__()

        self.cfg = cfg
        self.sparsity=sparsity
        self.coherence=coherence
        self.outcome_latent_model=outcome_model
        self.bias_latent_model=bias_model
        # assert self.bias_latent_model.vocab == self.bias_latent_model.vocab
        # self.vocab=self.bias_latent_model.vocab
        # self.criterion = nn.NLLLoss(reduction='none')

    @property
    def z(self):
        return self.outcome_model.generator.z

    @property
    def z_layer(self):
        return self.outcome_model.generator.z_layer

    @property
    def z_dists(self):
        return self.outcome_model.generator.z_dists
    def generate_rationale(self,x):
        mask = (x != 1)  # [B,T]
        _, z_outcome, _ = self.outcome_latent_model(x, mask)
        _, z_bias, _ = self.bias_latent_model(x, mask)
        self.z=z_outcome
        return z_outcome, z_bias
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
        bias_prediction, z_bias, z_bias_logits = self.bias_latent_model(x)

        outcome_prediction, z_outcome,z_outcome_logits = self.outcome_latent_model(x)
        self.z_bias=z_bias
        self.z_outcome=z_outcome
            
        return outcome_prediction, z_outcome, z_outcome_logits, bias_prediction, z_bias, z_bias_logits
    def get_bias_prediction(self, x):
        mask = (x != 1)  # [B,T]

        bias_prediction = self.bias_latent_model.classifier(x, mask, self.z_outcome)
        return bias_prediction
    def get_mean(self):
        return torch.mean(self.bias_model.generator.z_logits[0])
    def get_sigma(self):
        return torch.std(self.bias_model.generator.z_logits[0])
    def get_penality(self, mask=None, BIAS_THRED=0.):
        # compute energy constraint on z:
        # if BIAS_THRED_STRATEGY is 'mean':
        #     bias_threshold= self.get_mean()
        # elif BIAS_THRED_STRATEGY is '-1 sigma':
        #     bias_threshold = self.get_mean()-self.get_sigma()
        # elif BIAS_THRED_STRATEGY is 'hyperparameter':
        #     bias_threshold=BIAS_THRED
        # else:
        #     raise ValueError("no bias constraint strategy")
        
        

        # ================bias========================
        bias_z_dists = self.bias_latent_model.z_dists

        if len(bias_z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf_0_5 = bias_z_dists[0].cdf(0.5)
            raise NotImplementedError
        else:
            if self.outcome_latent_model.strategy == 2:
                cdf_0_5 = []
                for t in range(len(bias_z_dists)):
                    cdf_t = bias_z_dists[t].cdf(0.5)
                    cdf_0_5.append(cdf_t)
                cdf_0_5 = torch.stack(cdf_0_5, dim=1)
                bias_p0 = cdf_0_5
            else:
                pdf0 = []
                for t in range(len(bias_z_dists)):
                    pdf_t = bias_z_dists[t].pdf(0.)
                    pdf0.append(pdf_t)
                pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
                bias_p0 = pdf0

        bias_p0 = bias_p0.squeeze(-1)
        bias_p0 = torch.where(mask, bias_p0, bias_p0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        bias_p1 = 1. - bias_p0  # [B, T]
        bias_p1 = torch.where(mask, bias_p1, bias_p1.new_zeros([1]))

        # ================outcome========================
        outcome_z_dists = self.outcome_latent_model.z_dists

        if len(outcome_z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf_0_5 = outcome_z_dists[0].cdf(0.5)
            raise NotImplementedError
        else:
            if self.outcome_latent_model.strategy == 2:
                cdf_0_5 = []
                for t in range(len(outcome_z_dists)):
                    cdf_t = outcome_z_dists[t].cdf(0.5)
                    cdf_0_5.append(cdf_t)
                cdf_0_5 = torch.stack(cdf_0_5, dim=1)
                outcome_p0 = cdf_0_5
            else:
                pdf0 = []
                for t in range(len(outcome_z_dists)):
                    pdf_t = outcome_z_dists[t].pdf(0.)
                    pdf0.append(pdf_t)
                pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
                outcome_p0 = pdf0

        outcome_p0 = outcome_p0.squeeze(-1)
        outcome_p0 = torch.where(mask, outcome_p0, outcome_p0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        outcome_p1 = 1. - outcome_p0  # [B, T]
        outcome_p1 = torch.where(mask, outcome_p1, outcome_p1.new_zeros([1]))

        # ================penalty========================
        # outcome_z_logits = self.outcome_model.generator.z_logits[0].squeeze(1).squeeze(-1)  # [B, T]
        # bias_z_logits = self.bias_model.generator.z_logits[0].squeeze(1).squeeze(-1) 

        # bias_threshold = 1 - BIAS_THRED
        # tmp=bias_p1-bias_threshold # [B, T]
        # energy_penalty=(tmp>0) * (outcome_p1+tmp)

        if self.cfg['debias_method'] == 'energy':

            # bias_energy_threshold = - math.log(BIAS_THRED + self.cfg['eps']) #! use 0.5 for now
            # tmp = - torch.log(bias_p0 + self.cfg['eps']) - bias_energy_threshold
            # # add mask
            # tmp = torch.where(mask, tmp, tmp.new_zeros([1]))

            # energy_penalty = (tmp > 0) * ( - torch.log(outcome_p0 + self.cfg['eps'])+tmp)

            bias_threshold = 1 - BIAS_THRED
            tmp = bias_p1 - bias_threshold  # [B, T]
            energy_penalty = (tmp > 0) * ( - torch.log(outcome_p0 + self.cfg['eps']) + tmp)

        else:
            bias_threshold = 1 - BIAS_THRED
            tmp=bias_p1-bias_threshold # [B, T]
            energy_penalty=(tmp > 0) * (outcome_p1+tmp)
        # "tmp" has no gradient for backpropagation, thus could be omitted in the line above

        # return energy_penalty.mean()
        return energy_penalty.sum()/len(torch.where(mask)[0])

    def get_loss(self, logits, bias_prediction, targets, mask=None, BIAS_THRED_STRATEGY="mean", BIAS_THRED=0.5, BIAS_WEIGHT=0.1, **kwargs):
    
        optional = {}
        selection = self.outcome_latent_model.selection
        lasso = self.outcome_latent_model.lasso

        loss_vec = self.outcome_latent_model.criterion(logits, targets)  # [B]

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item()  # [1]

        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # z = self.generator.z.squeeze()
        z_dists = self.outcome_latent_model.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf_0_5 = z_dists[0].cdf(0.5)
            raise NotImplementedError
        else:
            if self.outcome_latent_model.strategy == 2:
                cdf_0_5 = []
                for t in range(len(z_dists)):
                    cdf_t = z_dists[t].cdf(0.5)
                    cdf_0_5.append(cdf_t)
                cdf_0_5 = torch.stack(cdf_0_5, dim=1)
                p0 = cdf_0_5
            else:
                pdf0 = []
                for t in range(len(z_dists)):
                    pdf_t = z_dists[t].pdf(0.)
                    pdf0.append(pdf_t)
                pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
                p0 = pdf0

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
        if self.cfg['abs']:
            c0_hat = torch.abs(l0 - selection)
        else:
            c0_hat = l0 - selection

        # moving average of the constraint
        self.outcome_latent_model.c0_ma = self.outcome_latent_model.lagrange_alpha * self.outcome_latent_model.c0_ma + \
            (1 - self.outcome_latent_model.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.outcome_latent_model.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.outcome_latent_model.lambda0 = self.outcome_latent_model.lambda0 * torch.exp(self.outcome_latent_model.lagrange_lr * c0.detach())


        bias_penalty=self.get_penality(mask=mask, BIAS_THRED=BIAS_THRED)
        loss = ce + bias_penalty * BIAS_WEIGHT
        optional['bias_penalty']=bias_penalty.item()


        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.outcome_latent_model.lambda0.item()
            optional["lagrangian0"] = (self.outcome_latent_model.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        loss = loss + self.outcome_latent_model.lambda0.detach() * c0

        if lasso > 0.:
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
            self.outcome_latent_model.c1_ma = self.outcome_latent_model.lagrange_alpha * self.outcome_latent_model.c1_ma + \
                (1 - self.outcome_latent_model.lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.outcome_latent_model.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.outcome_latent_model.lambda1 = self.outcome_latent_model.lambda1 * torch.exp(
                self.outcome_latent_model.lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["target1"] = lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.outcome_latent_model.lambda1.item()
                optional["lagrangian1"] = (self.outcome_latent_model.lambda1 * c1_hat).item()

            loss = loss + self.outcome_latent_model.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.outcome_latent_model.z, mask, strategy=self.outcome_latent_model.strategy)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional

    def get_loss_rl(self, outcome_prediction, bias_prediction, target, BIAS_THRED_STRATEGY='mean', BIAS_THRED=None, mask=None, **kwargs):
        
        optional = {}
        # selection = self.selection
        # lasso = self.lasso

        # outcome_prediction, z_outcome, z_outcome_logits = outcome[0], outcome[1], outcome[2]
        
        loss_vec = self.criterion(outcome_prediction, target)  # [B]

        # main MSE loss for p(y | x,z)
        loss = loss_vec.mean()        # [1]
        optional["ce"] = loss.item()  # [1]

        # compute energy constraint on z:
        if BIAS_THRED_STRATEGY is 'mean':
            bias_threshold= self.get_mean()
        elif BIAS_THRED_STRATEGY is '-1 sigma':
            bias_threshold = self.get_mean()-self.get_sigma()
        elif BIAS_THRED_STRATEGY is 'hyperparameter':
            bias_threshold=BIAS_THRED
        else:
            raise ValueError("no bias constraint strategy")
        outcome_z_logits = self.outcome_model.generator.z_logits[0].squeeze(1).squeeze(-1)  # [B, T]
        bias_z_logits = self.bias_model.generator.z_logits[0].squeeze(1).squeeze(-1) 
        
        tmp=bias_z_logits-bias_threshold # [B, T]
        energy_penalty=(tmp>0) * (outcome_z_logits+tmp)
        
        loss += energy_penalty.mean()
        optional['bias_penalty']=energy_penalty.mean().item()

        # compute generator loss
        z = self.outcome_model.generator.z.squeeze(1).squeeze(-1)  # [B, T]

        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.outcome_model.generator.z, mask)
            optional["p1"] = num_1 / float(total)

        # get P(z = 0 | x) and P(z = 1 | x)
        if len(self.outcome_model.generator.z_dists) == 1:  # independent z
            m = self.outcome_model.generator.z_dists[0]
            logp_z0 = m.log_prob(0.).squeeze(2)  # [B,T], log P(z = 0 | x)
            logp_z1 = m.log_prob(1.).squeeze(2)  # [B,T], log P(z = 1 | x)
        else:  # for dependent z case, first stack all log probs
            logp_z0 = torch.stack(
                [m.log_prob(0.) for m in self.outcome_model.generator.z_dists], 1).squeeze(2)
            logp_z1 = torch.stack(
                [m.log_prob(1.) for m in self.outcome_model.generator.z_dists], 1).squeeze(2)

        # compute log p(z|x) for each case (z==0 and z==1) and mask
        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask, logpz, logpz.new_zeros([1]))

        # sparsity regularization
        zsum = z.sum(1)  # [B]
        zdiff = z[:, 1:] - z[:, :-1]
        zdiff = zdiff.abs().sum(1)  # [B]

        zsum_cost = self.sparsity * zsum.mean(0)
        optional["zsum_cost"] = zsum_cost.item()

        zdiff_cost = self.coherence * zdiff.mean(0)
        optional["zdiff_cost"] = zdiff_cost.mean().item()

        sparsity_cost = zsum_cost + zdiff_cost
        optional["sparsity_cost"] = sparsity_cost.item()

        cost_vec = loss_vec + zsum * self.sparsity + zdiff * self.coherence
        cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward

        obj = cost_vec.mean()  # MSE with regularizers = neg reward
        optional["obj"] = obj.item()

        # generator cost
        optional["cost_g"] = cost_logpz.item()

        # encoder cost
        optional["cost_e"] = loss.item()

        return loss + cost_logpz, optional

  