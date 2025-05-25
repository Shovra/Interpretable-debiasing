import argparse
from evaluation.train_discrim import train_discrim

parser = argparse.ArgumentParser(description='jigsaw')
parser.add_argument('--save_path', type=str)
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

parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=-1)
parser.add_argument('--save_every', type=int, default=-1)

parser.add_argument('--dropout', type=float, default=0.5)

parser.add_argument('--layer', choices=["lstm"], default="lstm")
parser.add_argument('--train_embed', action='store_false', dest='fix_emb')


parser.add_argument("--eps", default=1e-6, type=float)
parser.add_argument("--mode", default='train')
parser.add_argument('--seed', default=0, type=int)

# misc
parser.add_argument('--word_vectors', type=str,
                    default='glove.840B.300d.sst.txt')
args = parser.parse_args()
if args.mode == 'train':
    train_discrim(args)
