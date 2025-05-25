import os
import sys
import time
from collections import OrderedDict
import json
import argparse


import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from debias_models.common.util import make_kv_string
from debias_models.energy_model.vocabulary import Vocabulary
from debias_models.energy_model.models.model_helpers import build_model
from debias_models.energy_model.util import get_args, jigsaw_reader, \
    prepare_minibatch, get_minibatch, load_glove, print_parameters, \
    initialize_model_, get_device
from debias_models.energy_model.evaluate import evaluate_single

device = get_device()
print("device:", device)


def test(cfg):
    cfg = vars(cfg)

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    #gender counter: Counter({0: 66319, 1: 53252, 2: 3929, 3: 1571})
    # frequency [0.53025082, 0.42577416, 0.03141416, 0.01256087]
    print("Loading data")
    train_data = list(jigsaw_reader("/data/wangyu/debias/biobias/train.json", label_field=cfg["label"]))
    dev_data = list(jigsaw_reader("/data/wangyu/debias/biobias/dev.json", label_field=cfg["label"]))
    test_data = list(jigsaw_reader("/data/wangyu/debias/biobias/test.json", label_field=cfg["label"]))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    vocab = Vocabulary()  # populated by load_glove
    glove_path = cfg["word_vectors"]
    vectors = load_glove(glove_path, vocab)


    iters_per_epoch = len(train_data) // cfg["batch_size"]

    if cfg["eval_every"] == -1:
        eval_every = iters_per_epoch
        print("Set eval_every to {}".format(iters_per_epoch))

    if cfg["num_iterations"] < 0:
        num_iterations = iters_per_epoch * -1 * cfg["num_iterations"]
        print("Set num_iterations to {}".format(num_iterations))

    example = dev_data[0]
    print("First train example:", example)
    print("First train example tokens:", example.tokens)
    print("First train example label:", example.label)

    if cfg["label"] == "label":
        # Map the sentiment labels 0-4 to a more readable form (and the opposite)

        i2t = ["non-toxic", "toxic"]
        t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})
    elif cfg["label"] == "gender":
        i2t = ["male", "female", "trans", "other"]
        t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})
    # Build model
    model = build_model(cfg["model"], vocab, t2i, cfg, use_adv=False)
                        # frequency=[0.53025082, 0.42577416, 0.03141416, 0.01256087])
    initialize_model_(model)

    with torch.no_grad():
        model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if cfg["fix_emb"]:
            print("fixed word embeddings")
            model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.  # padding zero

    model = model.to(device)

    # evaluate_single on test with best model
    print("# Loading best model")
    path = os.path.join(cfg["save_path"], f"{cfg['label']}_model.pt")
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print("No model found.")

    print("# Evaluating")
    dev_eval = evaluate_single(
        model, dev_data, batch_size=eval_batch_size,
        device=device)
    test_eval = evaluate_single(
        model, test_data, batch_size=eval_batch_size,
        device=device)

    print("dev {} test {}".format(
            make_kv_string(dev_eval),
            make_kv_string(test_eval)))