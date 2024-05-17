#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
# from network import *
import argparse

import network
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable( network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3_resnet50', choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")

    return parser

if __name__ == "__main__":

    # device = torch.device('cuda')
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = [513, 513]

    parser = argparse.ArgumentParser()
    opts = get_argparser().parse_args()

    opts.num_classes = 21

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    model.to(device)
    #model.eval()

    summary(model,(3,input_shape[0],input_shape[1]))

    dummy_input = torch.randn(1,3,input_shape[0],input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    flops, params = clever_format([flops,params],'%.3f')
    print('Total GFLOPS : %s' %(flops))
    print('Total params : %s' %(params))
