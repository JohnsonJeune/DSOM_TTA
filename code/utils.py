import torch

from data_processing import data_process_uci
from models.cnn import CNN_choose
from models.cnn_mix import MixCNN_choose
from models.adnn import adnn_choose

# All model parameters are placed on DEVICE (the first CUDA device when one is available)
from device import DEVICE


# model
def get_model(args):
    if args.model == 'cnn':
        model = CNN_choose(dataset = args.dataset, res=args.res)
    elif args.model == 'cnn_mix':
        model = MixCNN_choose(dataset = args.dataset, res=args.res)
    elif args.model == 'adnn':
        model = adnn_choose(dataset = args.dataset, res=args.res)
    else:
        raise ValueError("not exist this model: %r" % (args.model,))
    return model.to(DEVICE)



def get_dataset(args):
    if args.dataset != 'uci':
        raise ValueError("this project only keeps the uci dataset, current --dataset=%r" % (args.dataset,))
    source_loader, target_loader = data_process_uci.prep_domains_ucihar(
        args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(0.5*args.len_sw))
    return source_loader, target_loader
