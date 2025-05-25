#!/usr/bin/env python

from debias_models.energy_model.models.rl import RLModel
from debias_models.energy_model.models.latent import LatentRationaleModel

from debias_models.energy_model.models.latent import LatentRationaleModel

def build_model(model_type, vocab, t2i, cfg, frequency=None, use_adv=True):

    vocab_size = len(vocab.w2i)
    output_size = len(t2i)

    emb_size = cfg["embed_size"]
    hidden_size = cfg["hidden_size"]
    dropout = cfg["dropout"]
    layer = cfg["layer"]
    dependent_z = cfg.get("dependent_z", False)

    selection = cfg["selection"]
    lasso = cfg["lasso"]

    sparsity = cfg["sparsity"]
    coherence = cfg["coherence"]

    assert 0 < selection <= 1.0, "selection must be in (0, 1]"

    if model_type == "baseline":
        raise NotImplementedError("no baseline models currenly")
        # return Baseline(
        #     vocab_size, emb_size, hidden_size, output_size, vocab=vocab,
        #     dropout=dropout, layer=layer)
    elif model_type == "rl":
        return RLModel(
            vocab_size=vocab_size, emb_size=emb_size,
            hidden_size=hidden_size, output_size=output_size,
            vocab=vocab, dropout=dropout, layer=layer,
            dependent_z=dependent_z,
            sparsity=sparsity, coherence=coherence, frequency=frequency)
    elif model_type == "latent":
        # raise NotImplementedError("no latent models currenly")
        selection = cfg["selection"]
        lasso = cfg["lasso"]
        lagrange_alpha = cfg["lagrange_alpha"]
        lagrange_lr = cfg["lagrange_lr"]
        lambda_init = cfg["lambda_init"]
        lambda_min = cfg["lambda_min"]
        lambda_max = cfg["lambda_max"]
        strategy = cfg['strategy']
        return LatentRationaleModel(
            cfg=cfg, vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size,
            output_size=output_size, vocab=vocab, dropout=dropout,
            dependent_z=dependent_z, layer=layer,
            selection=selection, lasso=lasso, lambda_min=lambda_min,
            lagrange_alpha=lagrange_alpha, lagrange_lr=lagrange_lr,
            lambda_init=lambda_init, strategy=strategy, frequency=frequency)
    else:
        raise ValueError("Unknown model")
