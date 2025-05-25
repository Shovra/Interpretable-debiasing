from logging import raiseExceptions
import os
import sys
import time
import numpy as np
from collections import OrderedDict
from collections import defaultdict
from debias_models.common.util import get_z_stats
from tqdm import tqdm
import json

import torch
import torch.optim
from debias_models.energy_model.models.energymodel import EnergyModel
from debias_models.energy_model.models.model_helpers import build_model
sys.path.append("/data2/zexue/debias_by_rationale")

from debias_models.common.util import make_kv_string
from debias_models.energy_model.vocabulary import Vocabulary
from debias_models.common.discriminator import Discriminator
from debias_models.energy_model.util import jigsaw_reader_adv, \
    get_minibatch, prepare_minibatch_adv, load_glove, \
     get_device
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import entropy
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
    print("You want the rationale for ", label, " from jigsaw")
    train_data=None
    dev_data=None
    
    train_data = list(jigsaw_reader_adv("../datasets/jigsaw_gender/data/train.jsonl"))
    dev_data = list(jigsaw_reader_adv("../datasets/jigsaw_gender/data/dev.jsonl"))
    test_data = list(jigsaw_reader_adv("../datasets/jigsaw_gender/data/test.jsonl"))
    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data

def load_biobias_data(label="label"):
    print("Loading data ...")
    print("You want the rationale for ", label, " from biobias")
    train_data=None
    dev_data=None
    
    
    train_data = list(jigsaw_reader_adv("../datasets/biasinbio/train.json"))
    dev_data = list(jigsaw_reader_adv("../datasets/biasinbio/valid.json"))
    test_data = list(jigsaw_reader_adv("../datasets/biasinbio/test.json"))
    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data


def load_rationale_model(cfg):
    
    rationale_ckpt=torch.load(cfg["rationale_model_path"])
    vocab = Vocabulary()  # populated by load_glove
    glove_path = cfg["word_vectors"]
    vectors = load_glove(glove_path, vocab)
    if cfg["rationale_model_path"].find("debias")!=-1: # debias model, needs two rationale extractors
        # load reference bias model
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
        
            bias_i2t = ["male", "female","trans", "other"]#["male", "female"]
            bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
        if cfg["dataset"]=="jigsaw":
            outcome_i2t = ["non-toxic", "toxic"]
            outcome_t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})
            
            bias_i2t = ["male", "female","trans", "other"]
            bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
   

        # bias_i2t = ["male", "female","trans", "other"]
        # bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
        
        bias_model = build_model(rationale_ckpt["cfg"]["model"], vocab, bias_t2i, rationale_ckpt["cfg"], use_adv=False)
        
        # outcome_i2t = ["non-toxic", "toxic"]
        # outcome_t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})
        if "outcome_latent_model.criterion.weight" in rationale_ckpt["state_dict"]:
            outcome_model=build_model(rationale_ckpt["cfg"]["model"], vocab, outcome_t2i, rationale_ckpt["cfg"], use_adv=False, frequency=rationale_ckpt["state_dict"]["outcome_latent_model.criterion.weight"].cpu().tolist())
        else:
            outcome_model=build_model(rationale_ckpt["cfg"]["model"], vocab, outcome_t2i, rationale_ckpt["cfg"], use_adv=False, frequency=None)
        model = EnergyModel(rationale_ckpt["cfg"], outcome_model, bias_model)
        model.load_state_dict(rationale_ckpt["state_dict"])
        model.to(device)
    elif cfg["label"] == "label":
        # Map the sentiment labels 0-4 to a more readable form (and the opposite)
        if "criterion.weight" in rationale_ckpt["state_dict"]:     
            frequency=rationale_ckpt["state_dict"]["criterion.weight"]
        else:
            frequency=None
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
            i2t = list(outcome_dict.keys())
            t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})
        else:
            i2t = ["non-toxic", "toxic"]
            # i2t = ["male", "female","trans", "other"]
            t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})
        model = build_model(rationale_ckpt["cfg"]["model"], vocab, t2i, rationale_ckpt["cfg"], use_adv=False, frequency=frequency)
        model.load_state_dict(rationale_ckpt["state_dict"])
        model.to(device)
    elif cfg["label"] == "gender":
        if "criterion.weight" in rationale_ckpt["state_dict"]:     
            frequency=rationale_ckpt["state_dict"]["criterion.weight"]
        else:
            frequency=None
        # i2t = ["non-toxic", "toxic"]
        if cfg["dataset"] == "biobias":
            i2t=["male", "female"]
            t2i=OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})
        else:
            i2t = ["male", "female","trans", "other"]
            t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))}) 
        model = build_model(rationale_ckpt["cfg"]["model"], vocab, t2i, rationale_ckpt["cfg"], use_adv=False, frequency=frequency )
        model.load_state_dict(rationale_ckpt["state_dict"])
        model.to(device)
    model.eval()
    return model
def load_discrim_model(cfg):
    ckpt=torch.load(cfg["rationale_model_path"])
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
            bias_i2t = ["male", "female","trans", "other"]
    if cfg["dataset"]=="jigsaw":
            outcome_i2t = ["non-toxic", "toxic"]            
            bias_i2t = ["male", "female","trans", "other"]
   

    outcome_t2i = OrderedDict({p: i for p, i in zip(outcome_i2t, range(len(outcome_i2t)))})
    
    bias_t2i = OrderedDict({p: i for p, i in zip(bias_i2t, range(len(bias_i2t)))}) 
    

    vocab_size = len(vocab.w2i)
    emb_size = ckpt["cfg"]["embed_size"]
    hidden_size = ckpt["cfg"]["hidden_size"]
    dropout = ckpt["cfg"]["dropout"]
    layer = ckpt["cfg"]["layer"] 
    if cfg["label"] =="label":
        output_size = len(outcome_t2i)
    elif cfg["label"] == "gender":
        output_size = len(bias_t2i)
    else:
        ValueError("Unknown value")
    if "criterion.weight" in ckpt["state_dict"]:     
        frequency=ckpt["state_dict"]["criterion.weight"]
    else:
        frequency=None
    discrim = Discriminator(cfg=ckpt["cfg"], 
                            vocab=vocab, 
                            vocab_size=vocab_size, 
                            emb_size=emb_size, 
                            hidden_size=hidden_size,
                            output_size=output_size, 
                            dropout=dropout,
                            layer=layer, 
                            frequency=frequency)
    # discrim.load_state_dict(ckpt["state_dict"])
    state_dict=ckpt["state_dict"]
    with torch.no_grad():
        if cfg["label"] == "label":
            discrim.embed.weight.copy_(state_dict['outcome_latent_model.embed.weight'])
            discrim.classifier.embed_layer[0].weight.copy_(state_dict['outcome_latent_model.classifier.embed_layer.0.weight'])
            discrim.classifier.enc_layer.lstm.weight_ih_l0.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.weight_ih_l0"]) 
            discrim.classifier.enc_layer.lstm.weight_hh_l0.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.weight_hh_l0"]) 
            discrim.classifier.enc_layer.lstm.bias_ih_l0.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.bias_ih_l0"]) 
            discrim.classifier.enc_layer.lstm.bias_hh_l0.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.bias_hh_l0"]) 
            discrim.classifier.enc_layer.lstm.weight_ih_l0_reverse.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.weight_ih_l0_reverse"]) 
            discrim.classifier.enc_layer.lstm.weight_hh_l0_reverse.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.weight_hh_l0_reverse"]) 
            discrim.classifier.enc_layer.lstm.bias_ih_l0_reverse.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.bias_ih_l0_reverse"]) 
            discrim.classifier.enc_layer.lstm.bias_hh_l0_reverse.copy_(state_dict["outcome_latent_model.classifier.enc_layer.lstm.bias_hh_l0_reverse"]) 
            discrim.classifier.output_layer[1].weight.copy_(state_dict["outcome_latent_model.classifier.output_layer.1.weight"]) 
            discrim.classifier.output_layer[1].bias.copy_(state_dict["outcome_latent_model.classifier.output_layer.1.bias"]) 
        if cfg["label"] == "gender":
        
            discrim.embed.weight.copy_(state_dict['bias_latent_model.embed.weight'])
            discrim.classifier.embed_layer[0].weight.copy_(state_dict['bias_latent_model.classifier.embed_layer.0.weight'])
            discrim.classifier.enc_layer.lstm.weight_ih_l0.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.weight_ih_l0"]) 
            discrim.classifier.enc_layer.lstm.weight_hh_l0.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.weight_hh_l0"]) 
            discrim.classifier.enc_layer.lstm.bias_ih_l0.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.bias_ih_l0"]) 
            discrim.classifier.enc_layer.lstm.bias_hh_l0.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.bias_hh_l0"]) 
            discrim.classifier.enc_layer.lstm.weight_ih_l0_reverse.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.weight_ih_l0_reverse"]) 
            discrim.classifier.enc_layer.lstm.weight_hh_l0_reverse.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.weight_hh_l0_reverse"]) 
            discrim.classifier.enc_layer.lstm.bias_ih_l0_reverse.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.bias_ih_l0_reverse"]) 
            discrim.classifier.enc_layer.lstm.bias_hh_l0_reverse.copy_(state_dict["bias_latent_model.classifier.enc_layer.lstm.bias_hh_l0_reverse"]) 
            discrim.classifier.output_layer[1].weight.copy_(state_dict["bias_latent_model.classifier.output_layer.1.weight"]) 
            discrim.classifier.output_layer[1].bias.copy_(state_dict["bias_latent_model.classifier.output_layer.1.bias"]) 

    discrim.to(device)
    discrim.eval()
    return discrim
def compute_kl(cls_scores_, faith_scores_):
        keys = list(cls_scores_.keys())
        cls_scores_ = [cls_scores_[k] for k in keys]
        faith_scores_ = [faith_scores_[k] for k in keys]
        return entropy(faith_scores_, cls_scores_)
def eval(cfg):
    cfg = vars(cfg)
    rationale_model=load_rationale_model(cfg)
    discrim_model=load_discrim_model(cfg)
    if "seed" in rationale_model.cfg:
        seed=  rationale_model.cfg["seed"]  
    else:
        seed=0
    set_seed(seed)

    print("# Loaded best Rationale/discriminator model, random seed is ", seed)
    if cfg["dataset"] == "jigsaw":
        train_data, dev_data, test_data=load_data(cfg["label"])
    if cfg["dataset"] == "biobias":
        train_data, dev_data, test_data = load_biobias_data(cfg["label"])
    batch_size=cfg["batch_size"]
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
   
    
    # outcome_i2t = ["non-toxic", "toxic"]
    # bias_i2t = ["male", "female","trans", "other"]
    if not os.path.exists(cfg["save_path"]):
            os.makedirs(cfg["save_path"])
    
    for key, data in {"test": test_data}.items():#{"train":train_data, "test": test_data}.items() :#{"test": test_data}.items():
        print("# Now evaulate ", key, "dataset .....")
        metric=defaultdict(float)
        outcome_probs=[]
        outcome_preds=[]

        outcome_sufficiency_probs=[]
        outcome_sufficiency_preds=[]

        outcome_comprehensiveness_probs=[]
        outcome_comprehensiveness_preds=[]

        label_list=[]
        bias_list=[]
        z_totals = defaultdict(float)
        z_totals['p0']= 0
        z_totals['pc'] =0
        z_totals['p1'] =0
        z_totals['total']=0
        
        save_metric_file=os.path.join(cfg["save_path"], key+'.json')
        if cfg["generate"]:
            save_rationale_file=os.path.join(cfg["save_path"], key+'.txt')

            print("Write results to ", save_rationale_file)
            f=open(save_rationale_file, "w", errors='surrogatepass')
        else:
            f=None
        for mb in tqdm(get_minibatch(data, batch_size=batch_size,  shuffle=False)):
            
            x, targets, bias_targets, reverse_map = prepare_minibatch_adv(mb, discrim_model.vocab, device=device)
            sort_idx=np.argsort(reverse_map)
            mask = (x != 1)
            batch_size = targets.size(0)
            if cfg["label"]=="label":
                labels=targets.cpu().detach().numpy()
            if cfg["label"] == "gender":
                labels=bias_targets.cpu().detach().numpy()
            with torch.no_grad():
                if cfg["rationale_model_path"].find("debias")!=-1:
                    _, z_outcome, z_outcome_logits, _, z_bias, z_bias_logits = rationale_model(x)  # forward pass
                else:
                    _, z_outcome,_ =rationale_model(x)
                    z_bias=None
                outcome_pred, outcome_prob=discrim_model.reference(x, mask, z=None)

                n0, nc, n1, nt = get_z_stats(z_outcome, mask)
                z_totals['p0'] += n0
                z_totals['pc'] += nc
                z_totals['p1'] += n1
                z_totals['total'] += nt

                z_sufficiency=z_outcome
                outcome_sufficiency_pred, outcome_sufficiency_prob = discrim_model.reference(x, mask, z_sufficiency)


                z_comprehensiveness=1-z_outcome
                outcome_comprehensiveness_pred, outcome_comprehensiveness_prob = discrim_model.reference(x, mask, z_comprehensiveness)

                outcome_probs+=outcome_prob.detach().cpu().numpy().tolist()
                outcome_preds+=outcome_pred.detach().cpu().numpy().tolist()

                outcome_sufficiency_probs+=outcome_sufficiency_prob.detach().cpu().numpy().tolist()
                outcome_sufficiency_preds+=outcome_sufficiency_pred.detach().cpu().numpy().tolist()

                outcome_comprehensiveness_probs+=outcome_comprehensiveness_prob.detach().cpu().numpy().tolist()
                outcome_comprehensiveness_preds+=outcome_comprehensiveness_pred.detach().cpu().numpy().tolist()

                label_list+=labels.tolist()
                bias_list+=bias_targets.cpu().detach().numpy().tolist()
                if cfg["generate"]:
                    for i in range(z_outcome.shape[0]):
                        d=mb[sort_idx[i]]
                        f.write("====Example: "+str(i) + ", Bias "+ bias_i2t[d.adv_label] + ", Label "+ outcome_i2t[d.label] +"\n")
                        f.write("*Original: "+ " ".join(d.tokens)+"\n")
                        if z_bias !=None:
                            bias_rationale=[d.tokens[j] if z_bias[i][j]==1.0 else "_" for j in range(len(z_bias[i]))]
                            f.write("*Bias Rationale: " + " ".join(bias_rationale)+"\n")
                        final_rationale=[d.tokens[j] if z_outcome[i][j]==1.0 else "_" for j in range(len(z_outcome[i]))]
        
                        f.write("*Debiased Rationale: " + " ".join(final_rationale)+"\n")
        if cfg["generate"]:
            print("## finish generation for ", key)

            f.close()
    
        if cfg["sufficiency"]:
            sufficiency_scores = [outcome_probs[idx][label_list[idx]] - outcome_sufficiency_probs[idx][label_list[idx]] for idx in range(len(outcome_preds))]
            sufficiency_score = np.average(sufficiency_scores)
            print("## finish sufficiency score for ", key)

        else:
            sufficiency_score = None
            sufficiency_scores = None
        
        if cfg["comprehensiveness"]:
            comprehensiveness_scores = [outcome_probs[idx][label_list[idx]] - outcome_comprehensiveness_probs[idx][label_list[idx]] for idx in range(len(outcome_preds))]
            comprehensiveness_score = np.average(comprehensiveness_scores)
            print("## finish comprehensiveness score for ", key)

        else:
            comprehensiveness_scores = None
            comprehensiveness_score = None

        if cfg["comprehensiveness"]:
            comprehensiveness_entropies = [entropy(list(outcome_probs[idx])) - entropy(list(outcome_comprehensiveness_probs[idx])) for idx in range(len(outcome_preds))]
            comprehensiveness_entropy = np.average(comprehensiveness_entropies)
            print("## finish comprehensiveness entropy for ", key)

            #comprehensiveness_kl = np.average(list(compute_kl(outcome_probs[idx], outcome_comprehensiveness_probs[idx]) for idx in range(len(x))))
        else:
            comprehensiveness_entropies = None
            #comprehensiveness_kl = None
            comprehensiveness_entropy = None

        if cfg["sufficiency"]:
            sufficiency_entropies = [entropy(list(outcome_probs[idx])) - entropy(list(outcome_sufficiency_probs[idx])) for idx in range(len(outcome_preds))]
            sufficiency_entropy = np.average(sufficiency_entropies)
            print("## finish sufficiency entropy for ", key)

            #sufficiency_kl = np.average(list(compute_kl(outcome_probs[idx], outcome_sufficiency_probs[idx]) for idx in range(len(x))))
        else:
            sufficiency_entropies = None
            #sufficiency_kl = None
            sufficiency_entropy = None

        
        acc=accuracy_score(label_list, outcome_preds)
        macro=f1_score(label_list, outcome_preds, average='macro')
        micro=f1_score(label_list, outcome_preds, average='micro')
        print("## finish acc and f1 for ", key)

        sufficiency_acc=accuracy_score(label_list, outcome_sufficiency_preds)
        sufficiency_macro=f1_score(label_list, outcome_sufficiency_preds, average='macro')
        sufficiency_micro=f1_score(label_list, outcome_sufficiency_preds, average='micro')
        print("## finish sufficiency acc and f1 for ", key)
        comprehensiveness_acc=accuracy_score(label_list, outcome_comprehensiveness_preds)
        comprehensiveness_macro=f1_score(label_list, outcome_comprehensiveness_preds, average='macro')
        comprehensiveness_micro=f1_score(label_list, outcome_comprehensiveness_preds, average='micro')
        print("## finish comprehensiveness acc and f1 for ", key)

        # z scores
        z_totals['total'] += 1e-9
        for k, v in z_totals.items():
            if k != "total":
                metric[k] = v / z_totals["total"]

        if "p0" in metric:
            metric["selected"] = 1 - metric["p0"]
        print("## finish selection")

        metric["acc"]=acc
        metric["macro f1"]=macro
        metric["micro f1"]=micro
        
        metric["sufficiency acc"]=sufficiency_acc
        metric["sufficiency macro f1"]=sufficiency_macro
        metric["sufficiency micro f1"]=sufficiency_micro

        metric["comprehensiveness acc"]=comprehensiveness_acc
        metric["comprehensiveness macro f1"]=comprehensiveness_macro
        metric["comprehensiveness micro f1"]=comprehensiveness_micro

        metric["sufficiency scores"]=sufficiency_score        
        metric["sufficiency entropy"]=sufficiency_entropy

        metric["comprehensiveness scores"]=comprehensiveness_score        
        metric["comprehensiveness entropy"]=comprehensiveness_entropy

        print("##### evaluation result on ", key, "#####")
        
        for k, v in metric.items():
            print("* ", k, " : ", v)

        with open(save_metric_file, "w") as f:
            json.dump(metric, f)
        
        print("## save metric to file ", save_metric_file)

        if cfg["save_predictions"]:
            # save_prediction=os.path.join(cfg["save_path"], key+'_pred.json')
            with open(os.path.join(cfg["save_path"], key+'_outcome_pred.json'), "w") as f:
                json.dump(outcome_preds, f)
            with open(os.path.join(cfg["save_path"], key+'_gender_labels.json'), "w") as f:
                json.dump(bias_list, f)
            with open(os.path.join(cfg["save_path"], key+'_outcome_labels.json'), "w") as f:
                json.dump(label_list, f)
            with open(os.path.join(cfg["save_path"], key+'_outcome_sufficiency_pred.json'), "w") as f:
                json.dump(outcome_sufficiency_preds, f)
            with open(os.path.join(cfg["save_path"], key+'_outcome_comprehensive_pred.json'), "w") as f:
                json.dump(outcome_comprehensiveness_preds, f)