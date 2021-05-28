import argparse
import yaml
from torchvision import models
import os

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def make_dict(ll):
    return {k: v for (k, v) in ll}

class ParseKVToDictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs):
        if nargs != '+':
            raise argparse.ArgumentTypeError(f"ParseKVToDictAction can only be used for arguments with nargs='+' but instead we have nargs={nargs}")
        super(ParseKVToDictAction, self).__init__(option_strings, dest,
                nargs=nargs, type=type, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, option_string.lstrip('-'), make_dict(values))

def parse_input():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    subparsers_load = parser.add_subparsers(dest='load_method',required=True)

    parser_yaml = subparsers_load.add_parser("use-yaml-file")
    parser_yaml.add_argument('path',
                        help='path to YAML file')

    parser_arg = subparsers_load.add_parser("use-arguments")
    parser_arg.add_argument("experiment_name", type=str,
                help="Experiment name used to name log, model outputs.")
    parser_arg.add_argument('chunk_file_location',
                help='path to JSON file contains patches address')
    parser_arg.add_argument("--training_chunks", nargs="+", type=int, default=[0],
                help="Space separated number IDs specifying chunks to use for training.")
    parser_arg.add_argument("--validation_chunks", nargs="+", type=int, default=[1],
                help="Space separated number IDs specifying chunks to use for validation.")
    parser_arg.add_argument('patch_pattern',
                help='patterns of the stored patches')
    parser_arg.add_argument("subtypes", nargs='+', action=ParseKVToDictAction,
                help="space separated words describing subtype=groupping pairs for this study.")
    parser_arg.add_argument("--base_encoder", default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names))
    parser_arg.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension')
    parser_arg.add_argument('--epochs', default=5, type=int,
                        help='number of total epochs to run')
    parser_arg.add_argument('--normalize', action='store_true',
                        help='Normalize the dataset')
    parser_arg.add_argument('--use_cosine', action='store_true',
                        help='Using Cosine similarity in the loss function')
    parser_arg.add_argument('--num_patch_workers', default=4, type=int,
                        help='number of data loading workers')
    parser_arg.add_argument('-b', '--batch_size', default=256, type=int,
                        help='mini-batch size, this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser_arg.add_argument('--lr', '--learning_rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser_arg.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser_arg.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature')
    parser_arg.add_argument('--eval_every_n_steps', default=100, type=int,
                        help='Evaluation log every n steps')
    parser_arg.add_argument('--log_every_n_steps', default=100, type=int,
                        help='Log every n steps')

    #####################
    # TODO: use these flags
    # parser_arg.add_argument('--seed', default=None, type=int,
    #                     help='seed for initializing training. ')
    # parser_arg.add_argument('--disable-cuda', action='store_true',
    #                     help='Disable CUDA')
    # parser_arg.add_argument('--fp16-precision', action='store_true',
    #                     help='Whether or not to use 16-bit precision GPU training.')
    #####################

    args = parser.parse_args()
    return args

def parse_arguments():
    args = parse_input()
    if args.load_method == 'use-yaml-file':
        config = yaml.load(open(args.path, "r"), Loader=yaml.FullLoader)
        if not isinstance(config["weight_decay"], float):
            config["weight_decay"] = float(config["weight_decay"])
    elif args.load_method == 'use-arguments':
        config = vars(args)
    else:
        raise NotImplementedError(f"Loading from {args.load_method} is not implemented!")
    config["log_dir"] = os.path.join(config["log_dir"], f"{config['experiment_name']}")
    config["checkpoints"] = os.path.join(config["log_dir"], 'checkpoints')
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["checkpoints"], exist_ok=True)
    return config


if __name__ == "__main__":
    # print(model_names)
    print(parse_arguments())
