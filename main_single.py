import argparse
from debias_models.energy_model.train_single import train
from debias_models.energy_model.test import test

parser = argparse.ArgumentParser(description='jigsaw')
parser.add_argument('--save_path', type=str, default='results/')
parser.add_argument('--outcome_model_path', type=str, default='latent_outcome/model.pt')
parser.add_argument('--bias_model_path', type=str, default='latent_bias/model.pt')
parser.add_argument('--adv_model_path', type=str, default='adv/model.pt')
parser.add_argument('--resume_snapshot', type=str, default='')
parser.add_argument('--label', type=str, default="label", help="learn outcome label or bias label")

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

parser.add_argument('--model',
                    choices=["baseline", "rl", "attention",
                             "latent"],
                    default="baseline")
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
parser.add_argument('--abs', action='store_true', default=False,
                    help='whether to use abs on (c0 - selection)')
parser.add_argument("--eps", default=1e-6, type=float)
parser.add_argument("--mode", default='train')
parser.add_argument('--seed', default=0, type=int)

# misc
parser.add_argument('--word_vectors', type=str,
                    default='glove.840B.300d.sst.txt')
parser.add_argument("--strategy", type=int, default=1,
                    help='which deterministic strategy to choose')
args = parser.parse_args()
if args.mode == 'train':
    train(args)
else:
    test(args)