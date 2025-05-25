import os
import sys
import time
import numpy as np
from collections import OrderedDict
import json

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
# sys.path.append("/data2/zexue/debias_by_rationale")

from debias_models.energy_model.models.energymodel import EnergyModel
from debias_models.common.util import make_kv_string
from debias_models.energy_model.vocabulary import Vocabulary
from debias_models.energy_model.models.model_helpers import build_model
from debias_models.energy_model.util import get_args, jigsaw_reader_adv, \
    get_minibatch, prepare_minibatch_adv, load_glove, print_parameters, \
    initialize_model_, get_device
from debias_models.energy_model.evaluate import evaluate, generate_rationale

device = get_device()
print("device:", device)

def set_seed(seed):
    if seed == -1:
        seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(label="label"):
    print("Loading data ...")
    print("You want the rationale for ", label)
    
    train_data = list(jigsaw_reader_adv("../../datasets/jigsaw_gender/data/train.jsonl"))
    dev_data = list(jigsaw_reader_adv("../../datasets/jigsaw_gender/data/dev.jsonl"))
    test_data = list(jigsaw_reader_adv("../../datasets/jigsaw_gender/data/test.jsonl"))
    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data
def load_biobias_data(label="label"):
    print("Loading data ...")
    print("You want the rationale for ", label)
    
    train_data = list(jigsaw_reader_adv("../../datasets/biasinbio/train.json"))
    dev_data = list(jigsaw_reader_adv("../../datasets/biasinbio/valid.json"))
    test_data = list(jigsaw_reader_adv("../../datasets/biasinbio/test.json"))
    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data

def train(cfg):
    """
    Main training loop.
    """

    # cfg = get_args()
    cfg = vars(cfg)

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, v))

    set_seed(cfg['seed'])

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    print("Loading data")
    if cfg["dataset"]=="biobias":
        train_data, dev_data, test_data=load_biobias_data(cfg["label"]) #load_data(cfg["label"])
    if cfg["dataset"]=="jigsaw":
     train_data, dev_data, test_data=load_data(cfg["label"]) #load_data(cfg["label"])
    # if os.path.exists("/data/wangyu"):
    #     train_data = list(jigsaw_reader_adv("/data/wangyu/debias/biobias/train.json"))
    #     dev_data = list(jigsaw_reader_adv("/data/wangyu/debias/biobias/dev.json"))
    #     test_data = list(jigsaw_reader_adv("/data/wangyu/debias/biobias/test.json"))

    # else:
    #     train_data = list(jigsaw_reader_adv("/data2/zexue/FRESH/Datasets/jigsaw_gender/data/train.jsonl"))
    #     dev_data = list(jigsaw_reader_adv("/data2/zexue/FRESH/Datasets/jigsaw_gender/data/dev.jsonl"))
    #     test_data = list(jigsaw_reader_adv("/data2/zexue/FRESH/Datasets/jigsaw_gender/data/test.jsonl"))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

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

    writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard

    vocab = Vocabulary()  # populated by load_glove
    glove_path = cfg["word_vectors"]
    vectors = load_glove(glove_path, vocab)

    # outcome_i2t = [str(i) for i in range(33)]
    # outcome_t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})

    # bias_i2t = ["male", "female","trans", "other"]
    # bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
    
    if cfg["dataset"]=="biobias":
        outcome_dict={'professor': 0,
                        'paralegal': 1,
                        'attorney': 2,
                        'journalist': 3,
                        'comedian': 4,
                        'nurse': 5,
                        'physician': 6,
                        'personal_trainer': 7,
                        'painter': 8,
                        'surgeon': 9,
                        'teacher': 10,
                        'pastor': 11,
                        'architect': 12,
                        'composer': 13,
                        'dietitian': 14,
                        'photographer': 15,
                        'psychologist': 16,
                        'dentist': 17,
                        'model': 18,
                        'poet': 19,
                        'software_engineer': 20,
                        'massage_therapist': 21,
                        'filmmaker': 22,
                        'accountant': 23,
                        'rapper': 24,
                        'chiropractor': 25,
                        'dj': 26,
                        'interior_designer': 27,
                        'landscape_architect': 28,
                        'yoga_teacher': 29,
                        'magician': 30,
                        'real_estate_broker': 31,
                        'acupuncturist': 32}
        outcome_i2t = list(outcome_dict.keys())
        outcome_t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})
    
        bias_i2t = ["male", "female"]
        bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
    if cfg["dataset"]=="jigsaw":
        outcome_i2t = ["non-toxic", "toxic"]
        outcome_t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})
        
        bias_i2t = ["male", "female","trans", "other"]
        bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
    

    if cfg["label"] == "label":
    # Map the sentiment labels 0-4 to a more readable form (and the opposite)
        # i2t = ["non-toxic", "toxic"]
        i2t = outcome_i2t
        t2i = outcome_t2i
    elif cfg["label"] == "gender":
        i2t = bias_i2t
        t2i = bias_t2i
        
    # Build model
    # outcome_ckpt = torch.load(cfg["outcome_model_path"], map_location=device)
    # outcome_cfg = outcome_ckpt["cfg"]
    # outcome_model = build_model(outcome_cfg["model"], vocab, outcome_t2i, outcome_cfg, use_adv=False)
    # outcome_model.load_state_dict(outcome_ckpt["state_dict"])
    # outcome_model.to(device)
    
    # load bias model as reference
    bias_ckpt = torch.load(cfg["bias_model_path"], map_location=device)
    bias_cfg = bias_ckpt["cfg"]
    bias_model = build_model(bias_cfg["model"], vocab, bias_t2i, bias_cfg, use_adv=False)
    bias_model.load_state_dict(bias_ckpt["state_dict"])
    bias_model.to(device)

    outcome_model=build_model(cfg["model"], vocab, outcome_i2t, cfg, use_adv=False, frequency=[0.85483445, 0.14516555] if cfg['reweight'] else None)
    model = EnergyModel(cfg, outcome_model, bias_model)
    # set reference models
    with torch.no_grad():
        model.outcome_latent_model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if cfg["fix_emb"]:
            print("fixed outcome word embeddings")
            model.outcome_latent_model.embed.weight.requires_grad = False
        model.outcome_latent_model.embed.weight[1] = 0.  # padding zero
    
    for para in model.bias_latent_model.parameters():
        para.requires_grad = False

    optimizer = Adam(model.outcome_latent_model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_decay"], patience=cfg["patience"],
        verbose=True, cooldown=cfg["cooldown"], threshold=cfg["threshold"],
        min_lr=cfg["min_lr"])

    # print model
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    losses = []
    accuracies = []
    best_eval = 1.0e9
    best_iter = 0
    # print model
    model = model.to(device)
    print(model)
    print_parameters(model)
    
    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            epoch = iter_i // iters_per_epoch

            model.train()
            x, targets, bias_targets, _ = prepare_minibatch_adv(batch, model.bias_latent_model.vocab, device=device)
            mask = (x != 1)

            outcome_prediction, z_outcome, z_outcome_logits, bias_prediction, z_bias, z_bias_logits = model(x)  # forward pass

            loss, loss_optional = model.get_loss(outcome_prediction, 
                                                bias_prediction, 
                                                targets, 
                                                mask=mask,
                                                BIAS_THRED_STRATEGY="hyperparameter",
                                                BIAS_THRED=cfg['bias_thred'],
                                                BIAS_WEIGHT=cfg['bias_weight']
                                                )
            model.zero_grad()  # erase previous gradients

            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.outcome_latent_model.parameters(),
                                           max_norm=cfg["max_grad_norm"])
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:

                train_loss = train_loss / print_every
                writer.add_scalar('train/loss', train_loss, iter_i)
                for k, v in loss_optional.items():
                    writer.add_scalar('train/'+k, v, iter_i)

                print_str = make_kv_string(loss_optional)
                min_elapsed = (time.time() - start) // 60
                print("Epoch %r Iter %r time=%dm loss=%.4f %s" %
                      (epoch, iter_i, min_elapsed, train_loss, print_str))
                losses.append(train_loss)
                print_num = 0
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                dev_eval = evaluate(cfg, model, dev_data,
                                    batch_size=eval_batch_size, device=device)
                accuracies.append(dev_eval["outcome acc"])
                for k, v in dev_eval.items():
                    writer.add_scalar('dev/'+k, v, iter_i)

                print("# epoch %r iter %r: dev %s" % (
                    epoch, iter_i, make_kv_string(dev_eval)))

                # save best model parameters
                compare_score = dev_eval["loss"]
                if "obj" in dev_eval:
                    compare_score = dev_eval["obj"]

                scheduler.step(compare_score)  # adjust learning rate

                if (compare_score < (best_eval * (1-cfg["threshold"]))) and \
                        iter_i > (3 * iters_per_epoch):
                    print("***highscore*** %.4f" % compare_score)
                    best_eval = compare_score
                    best_iter = iter_i

                    for k, v in dev_eval.items():
                        writer.add_scalar('best/dev/' + k, v, iter_i)

                    if not os.path.exists(cfg["save_path"]):
                        os.makedirs(cfg["save_path"])

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "cfg": cfg,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    path = os.path.join(cfg["save_path"], "model.pt")
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("# Done training")

                # evaluate on test with best model
                print("# Loading best model")
                path = os.path.join(cfg["save_path"], "model.pt")
                if os.path.exists(path):
                    ckpt = torch.load(path)
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    print("No model found.")

                print("# Evaluating")
                dev_eval = evaluate(cfg,
                    model, dev_data, batch_size=eval_batch_size,
                    device=device)
                test_eval = evaluate(cfg,
                    model, test_data, batch_size=eval_batch_size,
                    device=device)

                print("best model iter {:d}: "
                      "dev {} test {}".format(
                        best_iter,
                        make_kv_string(dev_eval),
                        make_kv_string(test_eval)))

                sys.stdout.flush()

                # save result
                result_path = os.path.join(cfg["save_path"], "results.json")

                cfg["best_iter"] = best_iter

                for k, v in dev_eval.items():
                    cfg["dev_" + k] = v
                    writer.add_scalar('best/dev/' + k, v, iter_i)

                for k, v in test_eval.items():
                    print("test", k, v)
                    cfg["test_" + k] = v
                    writer.add_scalar('best/test/' + k, v, iter_i)

                writer.close()

                with open(result_path, mode="w") as f:
                    json.dump(cfg, f)
                
                return losses, accuracies



   

    # save result
    result_path = os.path.join(cfg["save_path"], "test_results.json")
    with open(result_path, mode="w") as f:
            json.dump(cfg, f)

    generate_rationale(
                    model, train_data, cfg, name="train", batch_size=eval_batch_size,
                    device=device)
    generate_rationale(
                    model, dev_data, cfg, name="dev",batch_size=eval_batch_size,
                    device=device)
    generate_rationale(
                    model, test_data, cfg, name="test",batch_size=eval_batch_size,
                    device=device)


if __name__ == "__main__":
    train()
