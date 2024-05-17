import torch
from torchsummary import summary
from thop import clever_format, profile

import torch.distributed as dist
import torch.utils.data.distributed

import argparse

import models as custom_models
import torchvision.models as models

import models.imagenet as customized_models


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dataloaders import *
from tensorboardX import SummaryWriter


default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +  ' | '.join(model_names) +   ' (default: resnet18)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')  
       

if __name__ == "__main__":

    device = torch.device('cuda')
    input_shape = [224,224]

    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    '''
    if args.distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
                                #world_size=args.world_size)
    '''

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = custom_models.__dict__[args.arch](width_mult=args.width_mult).to(device)


    #torch.cuda.set_device(device)
    model.cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[1], broadcast_buffers=False)

    '''
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        #model.cuda()
        torch.cuda.set_device(rank)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
    '''
    

    summary(model,(3,input_shape[0],input_shape[1]))

    dummy_input = torch.randn(1,3,input_shape[0],input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    flops, params = clever_format([flops,params],'%.3f')
    print('Total GFLOPS : %s' %(flops))
    print('Total params : %s' %(params))
