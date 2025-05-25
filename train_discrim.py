from logging import raiseExceptions
import os
import sys
import time
import numpy as np
from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
import collections
from tqdm import tqdm

import json
from nltk.tokenize import word_tokenize

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from debias_models.common.util import make_kv_string
from debias_models.energy_model.vocabulary import Vocabulary
from debias_models.common.discriminator import Discriminator
from debias_models.energy_model.util import jigsaw_reader_adv, \
    get_minibatch, prepare_minibatch_adv, load_glove, print_parameters, \
     get_device, advfilereader
from debias_models.energy_model.evaluate import  generate_rationale
from sklearn.metrics import accuracy_score, f1_score
AdvExample = namedtuple("AdvExample", ["tokens", "label", "token_labels", "adv_label"])

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
    
    train_data = list(jigsaw_reader_adv("../datasets/jigsaw_gender/train.jsonl"))
    dev_data = list(jigsaw_reader_adv("../datasets/jigsaw_gender/dev.jsonl"))
    test_data = list(jigsaw_reader_adv("../datasets/jigsaw_gender/test.jsonl"))
    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data
def load_biobias_data(label="label"):
    print("Loading data ...")
    print("You want the rationale for ", label)
    
    train_data = list(jigsaw_reader_adv("../datasets/biasinbio/train.json"))
    dev_data = list(jigsaw_reader_adv("../datasets/biasinbio/valid.json"))
    test_data = list(jigsaw_reader_adv("../datasets/biasinbio/test.json"))
    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data
def load_bold_data(label="label"):
    print("Loading data ...")
    print("You want the rationale for ", label)
    
    train_data = list(load_data_and_tok_train("../datasets/bold_new/train.json"))
    dev_data = list(load_data_and_tok_train("../datasets/bold_new/valid.json"))
    test_data = list(load_data_and_tok_train("../datasets/bold_new/test.json"))
    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data


def eval_discrim(model, data, batch_size=25, label="label", device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    totals=0
    predictions=[]
    true_labels=[]
    for mb in get_minibatch(data, batch_size=batch_size,  shuffle=False):
        x, targets, bias_targets, reverse_map = prepare_minibatch_adv(mb, model.vocab, device=device)
        mask = (x != 1)
        batch_size = targets.size(0)
        with torch.no_grad():

            logits = model(x) 
            pred=model.predict(logits) # forward pass
            predictions.append(pred.cpu().detach().numpy())

            if label=="label":

                loss, loss_optional = model.get_loss(logits, 
                                                targets, 
                                                mask=mask)
                true_labels.append(targets.cpu().detach().numpy())

                
            else:
                loss, loss_optional = model.get_loss(logits, 
                                                bias_targets, 
                                                mask=mask)
                true_labels.append(bias_targets.cpu().detach().numpy())

            if isinstance(loss, dict):
                loss = loss["main"]

            totals += loss.item() * batch_size


    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)

    acc=accuracy_score(flat_true_labels,flat_predictions)
    macro=f1_score(flat_true_labels, flat_predictions, average='macro')
    micro=f1_score(flat_true_labels, flat_predictions, average='micro')
    print("Test loss: ", 1.0*totals/len(flat_predictions),
        " , acc: ", acc,
        " , macro f1 score: ", macro, 
        " , micro f1 score: ", micro)


    return {"loss": 1.0*totals/len(flat_predictions), 
            "acc": acc,
            "macro f1": macro,
            "micro f1": micro}

    


def train_discrim(cfg):
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
    #for biobias
    if cfg["dataset"]=="biobias":
        train_data, dev_data, test_data=load_biobias_data(cfg["label"]) #load_data(cfg["label"])
    if cfg["dataset"]=="jigsaw":
     train_data, dev_data, test_data=load_data(cfg["label"]) #load_data(cfg["label"])
    if cfg["dataset"] =="bold":
        train_data, dev_data, test_data=load_bold_data()
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
    print("First train example bias:", example.adv_label)

    writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard

    vocab = Vocabulary()  # populated by load_glove
    glove_path = cfg["word_vectors"]
    vectors = load_glove(glove_path, vocab)

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
    if cfg["dataset"] =="bold":
        outcome_i2t = [None]
        outcome_t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})
        
        bias_i2t = ["European", "African", "Asian", "Latino"]
        bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
    
    vocab_size = len(vocab.w2i)
    emb_size = cfg["embed_size"]
    hidden_size = cfg["hidden_size"]
    dropout = cfg["dropout"]
    layer = cfg["layer"] 
    if cfg["label"] =="label":
        output_size = len(outcome_t2i)
    else:
        output_size = len(bias_t2i)
    

    if cfg["label"] == "label":
        model = Discriminator(cfg=cfg, 
                                vocab=vocab, 
                                vocab_size=vocab_size, 
                                emb_size=emb_size, 
                                hidden_size=hidden_size,
                                output_size=output_size, 
                                dropout=dropout,
                                layer=layer,
                                # frequency=[1.17, 6.88]) jigsaw
                                frequency=None) # biasbio
    else:
        model = Discriminator(cfg=cfg, 
                                vocab=vocab, 
                                vocab_size=vocab_size, 
                                emb_size=emb_size, 
                                hidden_size=hidden_size,
                                output_size=output_size, 
                                dropout=dropout,
                                layer=layer,
                                frequency=None)

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
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
    best_eval = 0#1.0e9
    best_iter = 0
    # print model
    model = model.to(device)
    print(model)
    print_parameters(model)
    
    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            epoch = iter_i // iters_per_epoch

            model.train()
            x, targets, bias_targets, _ = prepare_minibatch_adv(batch, model.vocab, device=device)
            mask = (x != 1)

            logits = model(x)  # forward pass
            if cfg["label"] == "label":
                loss, loss_optional = model.get_loss(logits, 
                                                targets, 
                                                mask=mask
                                                )
            else:
                loss, loss_optional = model.get_loss(logits, 
                                                bias_targets, 
                                                mask=mask
                                                )
            model.zero_grad()  # erase previous gradients

            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
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
                dev_eval = eval_discrim(model, dev_data,
                                    batch_size=eval_batch_size, label=cfg["label"], device=device)
                
                # save best model parameters
                compare_score = dev_eval["acc"]
                # if "obj" in dev_eval:
                #     compare_score = dev_eval["obj"]

                scheduler.step(compare_score)  # adjust learning rate

                # if (compare_score < (best_eval * (1-cfg["threshold"]))) and \
                #         iter_i > (3 * iters_per_epoch):
                if compare_score > best_eval:
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
                    torch.save(model, os.path.join(cfg["save_path"], "best_model.pt"))

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
                
                test_eval = eval_discrim(
                    model, test_data, label=cfg["label"],batch_size=eval_batch_size,
                    device=device)

                print("best model iter {:d}: "
                      "test {}".format(
                        best_iter,
                        make_kv_string(test_eval)))

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
def advfilereader_bold(path):
    """read jigsaw lines"""
    # with open(path, mode="r", encoding="utf-8") as f:
    #     for line in f:
    #         yield line.strip().replace("\\", "")
    x=[]
    y=[]
    adv_y=[]
    with open(path) as f:
        for i, line in enumerate(tqdm(f)):
                try:
                    d = eval(line)
                    
                    x.append(d["label"])
                    y.append(int(d["bias_id"]))
                    adv_y.append(int(d["bias_id"]))
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass
    toxic_frequency = collections.Counter(y)
    print("toxic label frequency", toxic_frequency)
    gender_frequency =  collections.Counter(adv_y)
    print("toxic label frequency", gender_frequency)
    return x,y, adv_y, toxic_frequency, gender_frequency



def load_data_and_tok_train( path, lower=True):
        lines, labels, adv_labels, _, _=advfilereader_bold(path)
        
        for idx in range(0,len(lines)):
            line=lines[idx]
            line = line.lower() if lower else line
            # line = re.sub("\\\\", "", line)  # fix escape

            tokens=word_tokenize(line)
            token_labels = [0]*len(tokens)
            
            label=labels[idx]
            adv_label=adv_labels[idx]
            # label = int(line[1])
            yield AdvExample(tokens=tokens, label=label, token_labels=token_labels, adv_label=adv_label)
       

def load_data_and_tok( gt_path="bold_new/test.json", 
                    gen_path="PPL_test_results/thred0_select0.5/generated.json",
                    lower=False):
        lines, labels, adv_labels, _, _=advfilereader_bold(gt_path)
        with open(gen_path, "r") as f:
            generated_list=json.load(f)
        
        for idx in range(0,len(lines)):
            line=generated_list[idx]
            line = line.lower() if lower else line
            # line = re.sub("\\\\", "", line)  # fix escape

            tokens=word_tokenize(line)
            token_labels = [0]*len(tokens)
            
            label=labels[idx]
            adv_label=adv_labels[idx]
            # label = int(line[1])
            yield AdvExample(tokens=tokens, label=label, token_labels=token_labels, adv_label=adv_label)
        
def test_single_file(args):
    model=torch.load(os.path.join( args.save_path, 'best_model.pt'))
    print("load data from ", args.save_path)
    
    test_data = list(load_data_and_tok(gt_path="bold_new/test.json",
    gen_path="PPL_test_results/gpt_result/gpt_probability_new/generated.json"))
    # gen_path="PPL_test_results/gpt_result/probabilitty/gt.json"))
    # gen_path="PL_test_results/gpt_result/gptfix/generated.json"))
    # gen_path="PPL_test_results/gpt_result/gptfix_bs/generated.json"))
    # gen_path="PPL_test_results/gpt_result/probabilitty/generated.json"))
    test_eval = eval_discrim(
    model, test_data, label="bias",batch_size=args.batch_size,
    device=device)
    print(test_eval)
    print("test {}".format(
                make_kv_string(test_eval)))
    
if __name__ == "__main__":
    train_discrim()
    