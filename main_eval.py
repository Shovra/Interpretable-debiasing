import argparse
from evaluation.eval_rationale import eval

parser = argparse.ArgumentParser(description='jigsaw')
parser.add_argument('--save_path', type=str, default='evaluate/')
parser.add_argument('--rationale_model_path', type=str, default='latent_debias/model.pt')
parser.add_argument('--discrim_model_path', type=str, default='latent_outcome/model.pt')
parser.add_argument('--resume_snapshot', type=str, default='')
parser.add_argument('--label', type=str, default="label")

parser.add_argument('--num_iterations', type=int, default=-30)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--eval_batch_size', type=int, default=512)

#eval options
parser.add_argument('--sufficiency', type=int, default=1)
parser.add_argument('--comprehensiveness', type=int, default=1)
parser.add_argument('--generate', type=int, default=1)


# misc
parser.add_argument('--word_vectors', type=str,
                    default='glove.840B.300d.sst.txt')
args = parser.parse_args()

eval(args)
