import torch.nn as nn
import torch
import torch.nn.functional as F

class Avg_max_channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(Avg_max_channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_avg = nn.Sequential(nn.Conv2d(in_channel, in_channel * ratio, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channel * ratio, in_channel,kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Softmax(dim=-1))
        self.mlp_max = nn.Sequential(nn.Conv2d(in_channel, in_channel * ratio,kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channel * ratio, in_channel,kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Softmax(dim=-1))

    def forward(self, x):
        # [b, c, n] ==> [b, c, 1]
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        # [b, c, 1] ==> [b, 1, c]
        x_avg = x_avg.transpose(-2, -1)
        x_max = x_max.transpose(-2, -1)
        x_avg_attn = self.mlp_avg(x_avg)
        x_max_attn = self.mlp_max(x_max)
        x_avg_attn = x_avg_attn.transpose(-2, -1)
        x_max_attn = x_max_attn.transpose(-2, -1)
        return x_avg_attn * x + x_max_attn * x

class cubic_attention(nn.Module):
    def __init__(self, dim, group=8, kernel=3) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))
    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta

class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w)
        
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out
