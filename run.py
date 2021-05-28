# import torch
# import torch.backends.cudnn as cudnn
from utils.parser import parse_arguments
# import utils.utils as utils
from simclr import SimCLR


def main():
    config = parse_arguments()
    # utils.manage_GPU(config)
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    # with torch.cuda.device(args.gpu_index):
    simclr = SimCLR(config)
    simclr.train()


if __name__ == "__main__":
    main()
