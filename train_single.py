import os
import pdb
import sys
import time
import numpy as np
from collections import OrderedDict
import json
import argparse
import sys
sys.path.append(r"E:\interpretable_debiasing-main\interpretable_debiasing-main\debias_models")

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
import nltk
nltk.data.path.append(r"C:\Users\NCB\AppData\Roaming\nltk_data")

# Test loading the tokenizer before running your script
nltk.word_tokenize("This is a test sentence.")

device = torch.device("cpu")  # Force CPU
print("⚠️ CUDA is not available. Running on CPU.")



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

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_folder = os.path.join(project_root, "datasets", "jigsaw_gender")

    train_data = list(jigsaw_reader(os.path.join(dataset_folder, "train.json")))
    dev_data = list(jigsaw_reader(os.path.join(dataset_folder, "dev.json")))
    print(f"Loaded {len(dev_data)} dev samples before filtering")
    test_data = list(jigsaw_reader(os.path.join(dataset_folder, "test.json")))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data
def load_biobias_data(label="label"):
    print("Loading data ...")
    print("You want the rationale for ", label)
    
    train_data = list(jigsaw_reader("../../datasets/biasinbio/train.json"))
    dev_data = list(jigsaw_reader("../../datasets/biasinbio/valid.json"))
    test_data = list(jigsaw_reader("../../datasets/biasinbio/test.json"))
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

    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    #gender counter: Counter({0: 66319, 1: 53252, 2: 3929, 3: 1571}) 
    # frequency [0.53025082, 0.42577416, 0.03141416, 0.01256087]
    if cfg["dataset"]=="biobias":
        train_data, dev_data, test_data=load_biobias_data(cfg["label"]) #load_data(cfg["label"])
    if cfg["dataset"]=="jigsaw":
     train_data, dev_data, test_data=load_data(cfg["label"]) #load_data(cfg["label"])
    if cfg["dataset"] =="bold":
        raise NotImplementedError
        
    # train_data = list(jigsaw_reader("../datasets/jigsaw_gender/train.json", label_field=cfg["label"]))
    # dev_data = list(jigsaw_reader("../datasets/jigsaw_gender/dev.json", label_field=cfg["label"]))
    # test_data = list(jigsaw_reader("../datasets/jigsaw_gender/test.json", label_field=cfg["label"]))

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
    
    # model = build_model(cfg["model"], vocab, t2i, cfg, use_adv=False, frequency=[0.53025082, 0.42577416, 0.03141416, 0.01256087] )
    model = build_model(cfg["model"], vocab, t2i, cfg, use_adv=False)
    initialize_model_(model)

    if len(cfg["resume_snapshot"])!=0:
        model_ckpt = torch.load(cfg["resume_snapshot"], map_location=device)
        cfg = model_ckpt["cfg"]
        model = build_model(cfg["model"], vocab, t2i, cfg, use_adv=False)
        model.load_state_dict(model_ckpt["state_dict"])
        model.to(device)
    # Build model
   
    with torch.no_grad():
        # Ensure vectors match model.embed.weight size
        embed_size = model.embed.weight.shape[0]
        if vectors.shape[0] != embed_size:
            print(f"⚠️ Mismatch: Model expects {embed_size} embeddings, but got {vectors.shape[0]}. Adjusting size...")
            vectors = vectors[:embed_size]  # Trim extra embeddings if needed

        model.embed.weight.data.copy_(torch.from_numpy(vectors))

        if cfg["fix_emb"]:
            print("fixed word embeddings")
            model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.  # padding zero

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_decay"], patience=cfg["patience"],
        verbose=True, cooldown=cfg["cooldown"], threshold=cfg["threshold"],
        min_lr=cfg["min_lr"])

    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    losses = []
    accuracies = []
    best_eval = 1.0e9
    best_iter = 0

    model = model.to(device)

    # print model
    print(model)
    print_parameters(model)

    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            epoch = iter_i // iters_per_epoch

            model.train()
            x, targets, _ = prepare_minibatch(batch, model.vocab, device=device)
            mask = (x != 1)

            logits,_,_ = model(x)  # forward pass

            loss, loss_optional = model.get_loss(logits, targets, mask=mask)
            model.zero_grad()  # erase previous gradients

            train_loss += loss.item()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                                max_norm=cfg["max_grad_norm"])
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
                dev_eval = evaluate_single(model, dev_data,
                                    batch_size=eval_batch_size, device=device)
                accuracies.append(dev_eval["acc"])
                for k, v in dev_eval.items():
                    writer.add_scalar('dev/'+k, v, iter_i)

                print("# epoch %r iter %r: dev %s \n\n" % (
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
                    path = os.path.join(cfg["save_path"], f"{cfg['label']}_model.pt")
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("# Done training")

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

                print("best model iter {:d}: "
                      "dev {} test {}".format(
                        best_iter,
                        make_kv_string(dev_eval),
                        make_kv_string(test_eval)))

                # save result
                result_path = os.path.join(cfg["save_path"], f"{cfg['label']}_results.json")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='jigsaw')
    parser.add_argument('--save_path', type=str, default='results/')
    parser.add_argument('--outcome_model_path', type=str, default='latent_outcome/model.pt')
    parser.add_argument('--bias_model_path', type=str, default='latent_bias/model.pt')
    parser.add_argument('--adv_model_path', type=str, default='adv/model.pt')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--label', type=str, default="label")

    parser.add_argument('--num_iterations', type=int, default=-30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--proj_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--cooldown', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=5.)

    parser.set_defaults(model="rl")  # ✅ Correct way to set a default
  # ✅ Use "rl" as the default model
    parser.add_argument('--model',
                    choices=["rl", "latent"],  # ✅ Only valid models
                    default="rl")  # ✅ Use "rl" as the default model

    parser.add_argument('--dist', choices=["", "hardkuma"],
                        default="")

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=-1)

    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--layer', choices=["lstm"], default="lstm")
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')

    parser.add_argument('--dependent-z', action='store_true',
                        help="make dependent decisions for z")

    # rationale settings for RL model
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--coherence', type=float, default=0.0)

    # rationale settings for HardKuma model
    parser.add_argument('--selection', type=float, default=0.3,
                        help="Target text selection rate for Lagrange.")
    parser.add_argument('--lasso', type=float, default=0.0)

    # lagrange settings
    parser.add_argument('--lagrange_lr', type=float, default=0.01,
                        help="learning rate for lagrange")
    parser.add_argument('--lagrange_alpha', type=float, default=0.99,
                        help="alpha for computing the running average")
    parser.add_argument('--lambda_init', type=float, default=1e-4,
                        help="initial value for lambda1")
    parser.add_argument('--lambda_min', type=float, default=1e-12,
                        help="initial value for lambda_min")
    parser.add_argument('--lambda_max', type=float, default=5.,
                        help="initial value for lambda_max")
    parser.add_argument('--dataset', type=str, default='jigsaw', 
                    help="Dataset to use. Options: 'jigsaw', 'biobias'")

    # misc
    parser.add_argument('--word_vectors', type=str,
                        default='glove.840B.300d.sst.txt')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    train(args)
