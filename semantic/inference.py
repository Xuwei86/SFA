import torch
import tqdm
from torch.backends import cudnn
import argparse
import numpy as np

from models import *

cudnn.benchmark = True
device = torch.device("cuda")
repetition = 500

dummy_input = torch.randn(1,3,224,224).to(device)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet)')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
global args, best_prec1
args = parser.parse_args()

# create model
if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.arch))
else:
    print("=> creating model '{}'".format(args.arch))

if args.arch == 'alexnet':
    model = alexnet(pretrained=args.pretrained).to(device)
elif args.arch == 'resnet18':
    model = resnet18(pretrained=args.pretrained).to(device)
elif args.arch == 'resnet34':
    model = resnet34(pretrained=args.pretrained).to(device)
elif args.arch == 'resnet50':
    model = resnet50(pretrained=args.pretrained).to(device)
elif args.arch == 'resnet101':
    model = resnet101(pretrained=args.pretrained).to(device)
elif args.arch == 'resnet152':
    model = resnet152(pretrained=args.pretrained).to(device)

print("Warm up ......\n")
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)

torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((repetition,1))

print('testing ... \n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetition)):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

avg = timings.sum()/repetition
fps = 1000 / avg
print('\n avg={}\n'.format(avg))
print('\n avg={}\n'.format(fps))


