import torch
import torch.nn as nn
import math

nano_o2s1 = 'tools/nano_sig1_weights.txt'
nano_o2s2 = 'tools/nano_sig2_weights.txt'
nano_eca = 'tools/nano_eca_weights.txt'
nano_sps1 = 'tools/nano_sps1_weights.txt'
nano_sps2 = 'tools/nano_sps2_weights.txt'
nano_sps18 = 'tools/nano_sps18_weights.txt'
nano_sps28 = 'tools/nano_sps28_weights.txt'
nano_sps1410 = 'tools/nano_sps14102_weights.txt'
nano_sps2410 = 'tools/nano_sps24102_weights.txt'
nano_sps1416 = 'tools/nano_sps1416_weights.txt'
nano_sps2416 = 'tools/nano_sps2416_weights.txt'
nano_sps1391 = 'tools/nano_sps1391_weights.txt'
nano_sps2391 = 'tools/nano_sps2391_weights.txt'

tiny_file21 = 'tools/tiny_sig2_weights.txt'
tiny_file22 = 'tools/tiny_sp2_weights.txt'
tiny_file23 = 'tools/tiny_sp220_weights.txt'

tiny_aug_cbam_file = 'tools/tiny_aug_cbam.txt'
tiny_no_aug_cbam_file = 'tools/tiny_no_aug_cbam.txt'
tiny_aug_se_file = 'tools/tiny_aug_se.txt'
tiny_no_aug_se_file = 'tools/tiny_no_aug_se.txt'
tiny_no_aug_eca_file = 'tools/tiny_no_aug_eca.txt'
tiny_aug_eca_file = 'tools/tiny_aug_eca.txt'

class modify_sigmoid20(torch.autograd.Function):
    @staticmethod
    def forward(self,inp):
        exp_x = torch.exp(-2*inp)
        result = 1/(1+exp_x)
        self.save_for_backward(result)
        return result
    @staticmethod
    def backward(self,grad_output):
        inp, = self.saved_tensors
        exp_x = torch.exp(-2*inp)
        return  grad_output * (2*exp_x/(1+exp_x)**2)
class ms20(nn.Module):
    def __int__(self):
        super().__int__()
    def forward(self,x):
        out = modify_sigmoid20.apply(x)
        return out


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # val1= [item.cpu().detach().numpy() for item in y[0]]
        # for i in val1:
        #     f = open(tiny_no_aug_se_file,'a')
        #     a = "%f,"%(i)
        #     f.write(str(a),)
        #     f.close
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=32, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)

        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # val1= [item.cpu().detach().numpy() for item in y[0][0]]
        # for i in val1:
        #     f = open(nano_eca_file,'a')
        #     a = "%f,"%(i)
        #     f.write(str(a),)
        #     f.close
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CA(nn.Module):
    def __init__(self, inp, groups=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=32):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class CA_1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_1d, self).__init__()

        # self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)
        self.conv_1x1 = nn.Conv1d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(channel // reduction)

        # self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        # self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_h = nn.Conv1d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv1d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        mult = torch.cat((x_h, x_w), 3).view(b, c, h+w)

        # x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(mult)))

        # x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 2)

        x_h = x_cat_conv_split_h
        x_w = x_cat_conv_split_w
        # s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        s_h = self.sigmoid_h(self.F_h(x_h)).view(b, c, h, 1)
        s_w = self.sigmoid_w(self.F_w(x_w)).view(b, c, 1, w)

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class CA_gn(nn.Module):
    def __init__(self, channel, reduction=24):
        super(CA_gn, self).__init__()

        mip = max(8,channel // reduction)

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=mip, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(channel // reduction)
        self.gn = nn.GroupNorm(mip, mip)
        self.F_h = nn.Conv2d(in_channels=mip, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=mip, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.gn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class CA2(nn.Module):
    def __init__(self,  channel1):
        super(CA2, self).__init__()

        ks1 = 3
        p1 = ks1 // 2
        self.conv1 = nn.Conv1d(channel1, channel1, kernel_size=ks1, padding=p1, bias=False)

        ks2 = 5
        p2 = ks2 // 2
        self.conv2 = nn.Conv1d(channel1, channel1, kernel_size=ks2, padding=p2, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # print("b={0} c={1} h={2}  w={3} ".format(b,c,h,w))

        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_h = x_h.view(b, c, h)
        x_w = x_w.view(b, c, h)

        x_h = self.sig(self.conv1(x_h))
        x_w = self.sig(self.conv1(x_w))
        x_h = x_h.view(b, c, h, 1 )
        x_w = x_w.view(b, c, 1, w)
        y = x_h * x_w

        return x * y

class CA22(nn.Module):
    def __init__(self,  channel1):
        super(CA22, self).__init__()

        ks1 = 3
        p1 = ks1 // 2
        self.conv1 = nn.Conv1d(channel1, channel1, kernel_size=ks1, padding=p1, bias=False)

        # ks2 = 5
        # p2 = ks2 // 2
        # self.conv2 = nn.Conv1d(channel1, channel1, kernel_size=ks2, padding=p2, bias=False)
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(channel1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_h = x_h.view(b, c, h)
        x_w = x_w.view(b, c, h)

        x_h = self.sig(self.conv1(x_h))
        x_w = self.sig(self.conv1(x_w))
        x_h = x_h.view(b, c, h, 1 )
        x_w = x_w.view(b, c, 1, w)
        y = x_h * x_w

        return x * y


class ELA81(nn.Module):
    def __init__(self,  channel, ks=3):
        super(ELA81, self).__init__()

        p0 = ks // 2
        p1 = (ks+2) // 2
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=ks, padding=p0, groups=channel, bias=False)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=ks+2, padding=p1, groups=channel, bias=False)
        # self.bn = nn.BatchNorm1d(channel)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.relu(self.conv1(x_h))
        x_w = self.relu(self.conv1(x_w))

        x_h = self.sig(self.conv2(x_h)).view(b, c, h, 1)
        x_w = self.sig(self.conv2(x_w)).view(b, c, 1, w)

        return x * x_h * x_w
class ELA83(nn.Module):
    def __init__(self,  channel, ks=3):
        super(ELA83, self).__init__()

        p0 = ks // 2
        p1 = (ks+2) // 2
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=ks, padding=p0, groups=channel, bias=False)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=ks+2, padding=p1, groups=channel, bias=False)
        self.gn = nn.GroupNorm(32,channel)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.relu(self.gn(self.conv1(x_h)))
        x_w = self.relu(self.gn(self.conv1(x_w)))

        x_h = self.sig(self.conv2(x_h)).view(b, c, h, 1)
        x_w = self.sig(self.conv2(x_w)).view(b, c, 1, w)

        return x * x_h * x_w

class ELA84(nn.Module):
    def __init__(self,  channel, ks=3):
        super(ELA84, self).__init__()

        p0 = ks // 2
        p1 = (ks+2) // 2
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=ks, padding=p0, groups=channel//8, bias=False)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=ks+2, padding=p1, groups=channel//8, bias=False)
        self.gn = nn.GroupNorm(32, channel)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.relu(self.gn(self.conv1(x_h)))
        x_w = self.relu(self.gn(self.conv1(x_w)))

        x_h = self.sig(self.conv2(x_h)).view(b, c, h, 1)
        x_w = self.sig(self.conv2(x_w)).view(b, c, 1, w)

        return x * x_h * x_w


class ELA72(nn.Module):
    def __init__(self,  channel, ks=7):
        super(ELA72, self).__init__()

        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=channel//8, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sig(self.gn(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.sig(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w

class ELA73(nn.Module):
    def __init__(self,  channel, ks=5):
        super(ELA73, self).__init__()

        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=1, bias=False)
        self.gn = nn.GroupNorm(32,channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sig(self.gn(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.sig(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w

class ELA702(nn.Module):
    def __init__(self,  channel, ks=5):
        super(ELA702, self).__init__()

        p = ks // 2
        self.conv = nn.Conv2d(channel, channel, kernel_size=ks, padding=p, groups=channel, bias=False)
        self.gn = nn.GroupNorm(32,channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_h = self.sig(self.gn(self.conv(x_h)))
        x_w = self.sig(self.gn(self.conv(x_w)))

        return x * x_h * x_w

class ELA6(nn.Module):
    def __init__(self,  channel, ks=5):
        super(ELA6, self).__init__()

        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=channel, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sig(self.conv(x_h)).view(b, c, h, 1)
        x_w = self.sig(self.conv(x_w)).view(b, c, 1, w)

        return x * x_h * x_w

class ELA10(nn.Module):
    def __init__(self,  channel, ks=5):
        super(ELA10, self).__init__()

        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=channel//8, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sig(self.conv(x_h)).view(b, c, h, 1)
        x_w = self.sig(self.conv(x_w)).view(b, c, 1, w)

        return x * x_h * x_w



class ELA2(nn.Module):
    def __init__(self,  channel, ks=3):
        super(ELA2, self).__init__()

        p = ks // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sig(self.conv(x_h)).view(b, c, h, 1)
        x_w = self.sig(self.conv(x_w)).view(b, c, 1, w)

        return x * x_h * x_w

class ELA3(nn.Module):
    def __init__(self,  channel, ratio=16, ks=3):
        super(ELA3, self).__init__()

        p = ks // 2
        self.conv1 = nn.Conv1d(channel, channel//ratio, kernel_size=ks, padding=p, bias=False)
        self.conv2 = nn.Conv1d(channel//ratio, channel, kernel_size=ks+2, padding=p+1, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sig(self.conv2(self.relu(self.conv1(x_h))))
        x_w = self.sig(self.conv2(self.relu(self.conv1(x_w))))

        x_h = x_h.view(b, c, h, 1)
        x_w = x_w.view(b, c, 1, w)

        return x * x_h * x_w

class mma1(nn.Module):
    def __init__(self, channel, ks=7, gamma=2, b=1):
        super(mma1, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.conv0 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        self.conv1 = nn.Conv1d(channel//4, channel//4, kernel_size=ks, padding=p, groups=1, bias=False)

        p1 = 3 if ks== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, ks, padding=p1, bias=False)

    def forward(self, x):

        y = torch.chunk(x, 4, dim=1)

        # channel attention
        b1, c1, h, w = y[1].shape

        xc = self.avg_pool(y[1]).view([b1, 1, c1])
        xc = self.conv0(xc)
        xc = self.sig(xc).view([b1, c1, 1, 1])
        xc = y[1] * xc

        # spatial0 attention
        b2, c2, h, w = y[2].shape

        x_h = torch.mean(y[2], dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(y[2], dim=2, keepdim=True).view(b2, c2, w)
        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)
        xs = y[2] * x_h * x_w

        # spatial1 attention
        b1, c1, h, w = y[3].shape
        avg_out = torch.mean(y[3], dim=1, keepdim=True)
        max_out, _ = torch.max(y[3], dim=1, keepdim=True)
        yn = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv2(yn).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = y[3] * xn

        # concatenate along channel axis
        out = torch.cat([y[0], xc, xs, xn], dim=1)

        return out

class mma2(nn.Module):
    def __init__(self, channel, ks=7, gamma=2, b=1):
        super(mma2, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.conv0 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        self.conv1 = nn.Conv1d(channel//4, channel//4, kernel_size=ks, padding=p, groups=channel//4, bias=False)

        p1 = 3 if ks== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, ks, padding=p1, bias=False)

    def forward(self, x):

        y = torch.chunk(x, 4, dim=1)

        # channel attention
        b1, c1, h, w = y[1].shape

        xc = self.avg_pool(y[1]).view([b1, 1, c1])
        xc = self.conv0(xc)
        xc = self.sig(xc).view([b1, c1, 1, 1])
        xc = y[1] * xc

        # spatial0 attention
        b2, c2, h, w = y[2].shape

        x_h = torch.mean(y[2], dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(y[2], dim=2, keepdim=True).view(b2, c2, w)
        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)
        xs = y[2] * x_h * x_w

        # spatial1 attention
        b1, c1, h, w = y[3].shape
        avg_out = torch.mean(y[3], dim=1, keepdim=True)
        max_out, _ = torch.max(y[3], dim=1, keepdim=True)
        yn = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv2(yn).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = y[3] * xn

        # concatenate along channel axis
        out = torch.cat([y[0], xc, xs, xn], dim=1)

        return out

class mma3(nn.Module):
    def __init__(self, channel, ks=7, gamma=2, b=1):
        super(mma3, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.conv0 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        self.conv1 = nn.Conv1d(channel//4, channel//4, kernel_size=ks, padding=p, groups=channel//16, bias=False)

        p1 = 3 if ks== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, ks, padding=p1, bias=False)

    def forward(self, x):

        y = torch.chunk(x, 4, dim=1)

        # channel attention
        b1, c1, h, w = y[1].shape

        xc = self.avg_pool(y[1]).view([b1, 1, c1])
        xc = self.conv0(xc)
        xc = self.sig(xc).view([b1, c1, 1, 1])
        xc = y[1] * xc

        # spatial0 attention
        b2, c2, h, w = y[2].shape

        x_h = torch.mean(y[2], dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(y[2], dim=2, keepdim=True).view(b2, c2, w)
        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)
        xs = y[2] * x_h * x_w

        # spatial1 attention
        b1, c1, h, w = y[3].shape
        avg_out = torch.mean(y[3], dim=1, keepdim=True)
        max_out, _ = torch.max(y[3], dim=1, keepdim=True)
        yn = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv2(yn).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = y[3] * xn

        # concatenate along channel axis
        out = torch.cat([y[0], xc, xs, xn], dim=1)

        return out

class mma4(nn.Module):
    def __init__(self, channel, ks=7, gamma=2, b=1):
        super(mma4, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.conv0 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        self.conv1 = nn.Conv1d(channel//2, channel//2, kernel_size=ks, padding=p, groups=channel//2, bias=False)

    def forward(self, x):

        y = torch.chunk(x, 4, dim=1)

        # channel attention
        b1, c1, h, w = y[1].shape

        xc = self.avg_pool(y[1]).view([b1, 1, c1])
        xc = self.conv0(xc)
        xc = self.sig(xc).view([b1, c1, 1, 1])
        xc = y[1] * xc

        # spatial0 attention
        ys = torch.cat([y[2], y[3]], dim=1)
        b2, c2, h, w = ys.shape

        x_h = torch.mean(ys, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(ys, dim=2, keepdim=True).view(b2, c2, w)
        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)
        xs = ys * x_h * x_w


        # concatenate along channel axis
        out = torch.cat([y[0], xc, xs], dim=1)

        return out

class SGE(nn.Module):

    def __init__(self, groups=8):
        super(SGE, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        from torch.nn.parameter import Parameter
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class sa_la(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, ks=7, groups=8, gamma=2, b=1):
        super(sa_la, self).__init__()
        from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False )

        p = ks // 2
        c = channel // (groups * 2)
        self.conv2 = nn.Conv1d(c, c, kernel_size=ks, padding=p, groups=1, bias=False)
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.sig = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        b1, c1, h, w = x_0.shape

        y = self.avg_pool(x_0).view([b1, 1, c1])
        y = self.conv1(y)
        y = self.sig(y).view([b1, c1, 1, 1])
        xn =  x_0 * y

        # spatial attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.gn(self.conv2(x_h))).view(b2, c2, h, 1)
        x_w = self.sig(self.gn(self.conv2(x_w))).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class sa_ela(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, ks=7, groups=16, gamma=2, b=1):
        super(sa_ela, self).__init__()
        from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        c = channel // (groups * 2)
        self.conv2 = nn.Conv1d(c, c, kernel_size=ks, padding=p, groups=1, bias=False)
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        b1, c1, h, w = x_0.shape

        y = self.avg_pool(x_0).view([b1, 1, c1])
        y = self.conv1(y)
        y = self.sig(y).view([b1, c1, 1, 1])
        xn = x_0 * y

        # spatial attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.gn(self.conv2(x_h))).view(b2, c2, h, 1)
        x_w = self.sig(self.gn(self.conv2(x_w))).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela2(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, ks=7, groups=32, gamma=2, b=1):
        super(sa_ela2, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        c = channel // (groups * 2)
        self.conv2 = nn.Conv1d(c, c, kernel_size=ks, padding=p, groups=1, bias=False)
        # self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        b1, c1, h, w = x_0.shape

        y = self.avg_pool(x_0).view([b1, 1, c1])
        y = self.conv1(y)
        y = self.sig(y).view([b1, c1, 1, 1])
        xn = x_0 * y

        # spatial attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv2(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv2(x_w)).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela20_t(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, ks=7, groups=32, gamma=2, b=1):
        super(sa_ela20_t, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        c = channel // (groups * 2)
        self.conv2 = nn.Conv1d(c, c, kernel_size=ks, padding=p, groups=1, bias=False)
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        b1, c1, h, w = x_0.shape

        y = self.avg_pool(x_0).view([b1, 1, c1])
        y = self.conv1(y)
        y = self.sig(y).view([b1, c1, 1, 1])
        xn = x_0 * y

        # spatial attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.gn(self.conv2(x_h))).view(b2, c2, h, 1)
        x_w = self.sig(self.gn(self.conv2(x_w))).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela3(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, ks=5, groups=32, gamma=2, b=1):
        super(sa_ela3, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        c = channel // (groups * 2)
        self.conv21 = nn.Conv1d(c, c, kernel_size=ks, padding=p, groups=1, bias=False)
        p = (ks-2) // 2
        c = channel // (groups * 2)
        self.conv20 = nn.Conv1d(c, c, kernel_size=ks-2, padding=p, groups=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        b1, c1, h, w = x_0.shape

        y = self.avg_pool(x_0).view([b1, 1, c1])
        y = self.conv1(y)
        y = self.sig(y).view([b1, c1, 1, 1])
        xn = x_0 * y

        # spatial attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv21(self.sig(self.conv20(x_h)))).view(b2, c2, h, 1)
        x_w = self.sig(self.conv21(self.sig(self.conv20(x_w)))).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out
class sa_ela10(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=32, kernel=7):
        super(sa_ela10, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        p1 = 3 if kernel== 7 else 1
        self.conv0 = nn.Conv2d(2, 1, kernel, padding=p1, bias=False)

        p2 = ks // 2
        c = channel // (groups * 2)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()
        # self.gn = nn.GroupNorm(16, channel)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # spatial1 attention
        b1, c1, h, w = x_0.shape
        avg_out = torch.mean(x_0, dim=1, keepdim=True)
        max_out, _ = torch.max(x_0, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv0(y).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = x_0 * xn


        # spatial2 attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela11(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=32, kernel=7):
        super(sa_ela11, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups

        p1 = 3 if kernel== 7 else 1
        self.conv0 = nn.Conv2d(2, 1, kernel, padding=p1, bias=False)

        p2 = ks // 2
        c = channel // (groups * 2)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()
        # self.gn = nn.GroupNorm(16, channel)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # spatial1 attention
        b1, c1, h, w = x_0.shape
        avg_out = torch.mean(x_0, dim=1, keepdim=True)
        max_out, _ = torch.max(x_0, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv0(y).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = x_0 * xn

        # spatial2 attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out

class sa_ela20(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=32, kernel=7):
        super(sa_ela20, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        p2 = ks // 2
        c = channel // (groups * 2)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()
        # self.gn = nn.GroupNorm(16, channel)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # spatial2 attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([x_0, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela21(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=32, kernel=7):
        super(sa_ela21, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        p2 = ks // 2
        c = channel // (groups * 2)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()
        # self.gn = nn.GroupNorm(32, c)
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # spatial2 attention

        b2, c2, h, w = x_1.shape

        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([x_0, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out

class sa_ela22(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=16):
        super(sa_ela22, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        p1 = 3 if ks== 7 else 1
        self.conv0 = nn.Conv2d(2, 1, ks, padding=p1, bias=False)

        p2 = ks // 2
        c = channel // (groups * 4)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()
        # self.gn = nn.GroupNorm(16, channel)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        y = torch.chunk(x, 4, dim=1)

        # spatial0 attention

        b2, c2, h, w = y[1].shape

        x_h = torch.mean(y[1], dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(y[1], dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)

        xs = y[1] * x_h * x_w

        # spatial1 attention
        b1, c1, h, w = y[3].shape
        avg_out = torch.mean(y[3], dim=1, keepdim=True)
        max_out, _ = torch.max(y[3], dim=1, keepdim=True)
        yn = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv0(yn).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = y[3] * xn

        # concatenate along channel axis
        out = torch.cat([y[0], xs, y[2], xn], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela23(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=16, gamma=2, b=1):
        super(sa_ela23, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p2 = ks // 2
        c = channel // (groups * 4)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()

        p1 = 3 if ks== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, ks, padding=p1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        y = torch.chunk(x, 4, dim=1)

        # channel attention
        b1, c1, h, w = y[1].shape

        xc = self.avg_pool(y[1]).view([b1, 1, c1])
        xc = self.conv0(xc)
        xc = self.sig(xc).view([b1, c1, 1, 1])
        xc = y[1] * xc

        # spatial0 attention
        b2, c2, h, w = y[2].shape

        x_h = torch.mean(y[2], dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(y[2], dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)

        xs = y[2] * x_h * x_w

        # spatial1 attention
        b1, c1, h, w = y[3].shape
        avg_out = torch.mean(y[3], dim=1, keepdim=True)
        max_out, _ = torch.max(y[3], dim=1, keepdim=True)
        yn = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv2(yn).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = y[3] * xn

        # concatenate along channel axis
        out = torch.cat([y[0], xc, xs, xn], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela232(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=16, gamma=2, b=1):
        super(sa_ela232, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p2 = ks // 2
        c = channel // (groups * 4)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()

        p1 = 3 if ks== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, ks, padding=p1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        y = torch.chunk(x, 4, dim=1)

        # channel attention
        b1, c1, h, w = y[1].shape

        xc = self.avg_pool(y[1]).view([b1, 1, c1])
        xc = self.conv0(xc)
        xc = self.sig(xc).view([b1, c1, 1, 1])
        xc = y[1] * xc

        # spatial0 attention
        b2, c2, h, w = y[2].shape

        x_h = torch.mean(y[2], dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(y[2], dim=2, keepdim=True).view(b2, c2, w)

        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)

        xs = y[2] * x_h * x_w

        # spatial1 attention
        b1, c1, h, w = y[3].shape
        avg_out = torch.mean(y[3], dim=1, keepdim=True)
        max_out, _ = torch.max(y[3], dim=1, keepdim=True)
        yn = torch.cat([avg_out, max_out], dim=1)
        xn = self.conv2(yn).expand([b1, c1, h, w])
        xn = self.sig(xn)
        xn = y[3] * xn

        # concatenate along channel axis
        out = torch.cat([y[0], xc, xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class sa_ela233(nn.Module):
    """Constructs a Channel Spatial Group module.
        Args:
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, ks=7, groups=32, gamma=2, b=1):
        super(sa_ela233, self).__init__()
        # from torch.nn.parameter import Parameter

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p2 = ks // 2
        c = channel // (groups * 4)
        self.conv1 = nn.Conv1d(c, c, kernel_size=ks, padding=p2, groups=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        y = torch.chunk(x, 4, dim=1)

        # channel attention
        b1, c1, h, w = y[1].shape

        xc = self.avg_pool(y[1]).view([b1, 1, c1])
        xc = self.conv0(xc)
        xc = self.sig(xc).view([b1, c1, 1, 1])
        xc = y[1] * xc

        # spatial0 attention
        b2, c2, h, w = y[3].shape

        x_h = torch.mean(y[3], dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(y[3], dim=2, keepdim=True).view(b2, c2, w)
        x_h = self.sig(self.conv1(x_h)).view(b2, c2, h, 1)
        x_w = self.sig(self.conv1(x_w)).view(b2, c2, 1, w)
        xs = y[3] * x_h * x_w

        # concatenate along channel axis
        out = torch.cat([y[0], xc, y[2], xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out

class SE_SA(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(SE_SA, self).__init__()
        mch = max(8, channel//reduction)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()
        # CB
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding2, bias=False)
        self.bn = nn.BatchNorm2d(channel)


    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        x_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        x_w = self.sig(self.conv11(x_w))
        F_ca = x_h * x_w   # b, c, h, w

        F_se = self.conv11(self.relu1(self.conv10(self.avg_pool(x) + self.max_pool(x))))
        F = F_ca + F_se

        return x * F



class CA_SA41(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3):
        super(CA_SA41, self).__init__()
        mch = max(8, channel//reduction)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        x_h = self.conv11(x_h.permute(0, 1, 3, 2))
        x_w = self.conv11(x_w)
        F1 = x_h * x_w   # b, c, h, w

        F2 = torch.mean(x, dim=1, keepdim=True)
        F2 = self.conv2(F2)  # b,1,h,w

        F = self.sig(F1 + F2)

        return x * F

class CA_SA42(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3):
        super(CA_SA42, self).__init__()
        mch = max(8,channel//reduction)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        x_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        x_w = self.sig(self.conv11(x_w))
        F1 = x_h * x_w   # b, c, h, w

        F2 = torch.mean(x, dim=1, keepdim=True)
        F2 = self.sig(self.conv2(F2))  # b,1,h,w

        F = F1 * F2

        return x * F

class CA_SA5(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3):
        super(CA_SA5, self).__init__()

        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(channel // reduction)
        self.conv11 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        x_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        x_w = self.sig(self.conv11(x_w))
        F1 = x_h * x_w   # b, c, h, w

        F2 = torch.mean(x, dim=1, keepdim=True)
        F2 = self.sig(self.bn(self.conv2(F2)))  # b,1,h,w

        F = F1 + F2

        return x * F

class CB_CA(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(CB_CA, self).__init__()
        # CA
        mch = max(8,channel//ratio)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        # CB
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接
        self.relu1 = nn.ReLU()

        # SA
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding2 = 3 if kernel_size == 7 else 1
        # self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        avg_out = self.conv11(self.relu(self.conv10(self.avg_pool(x))))
        max_out = self.conv11(self.relu(self.conv10(self.max_pool(x))))
        F_cb = self.sig(avg_out + max_out)   # b, c, 1, 1
        F = x * F_cb

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        s_w = self.sig(self.conv11(x_w))
        F_ca = s_h * s_w  # b, c, h, w

        out = F * F_ca

        return out

class CB_CA_SA2(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(CB_CA_SA2, self).__init__()
        # CA
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=channel // ratio, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(channel // ratio)
        self.conv11 = nn.Conv2d(in_channels=channel // ratio, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        # CB
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        avg_out = self.conv11(self.relu(self.conv10(self.avg_pool(x))))
        max_out = self.conv11(self.relu(self.conv10(self.max_pool(x))))
        F_cb = self.sig(avg_out + max_out)   # b, c, 1, 1
        F = x * F_cb

        F_sa = torch.mean(x, dim=1, keepdim=True)
        F_sa = self.sig(self.conv2(F_sa))  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        s_w = self.sig(self.conv11(x_w))
        F_ca = s_h * s_w  # b, c, h, w

        out = F_sa + F_ca

        F = F * out

        return F

class CB_CA_SA3(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(CB_CA_SA3, self).__init__()
        # CA
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=channel // ratio, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(channel // ratio)
        self.conv11 = nn.Conv2d(in_channels=channel // ratio, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        # CB
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        avg_out = self.conv11(self.relu(self.conv10(self.avg_pool(x))))
        max_out = self.conv11(self.relu(self.conv10(self.max_pool(x))))
        F_cb = self.sig(avg_out + max_out)   # b, c, 1, 1
        F = x * F_cb

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        F_sa = torch.cat([avg_out, max_out], dim=1)
        F_sa = self.sig(self.conv2(F_sa))  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        s_w = self.sig(self.conv11(x_w))
        F_ca = s_h * s_w  # b, c, h, w

        out = F_sa + F_ca

        F = F * out

        return F

class CB_CA_SA4(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(CB_CA_SA4, self).__init__()
        # CA
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=channel // ratio, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(channel // ratio)
        self.conv11 = nn.Conv2d(in_channels=channel // ratio, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        # CB
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        avg_out = self.conv11(self.relu(self.conv10(self.avg_pool(x))))
        max_out = self.conv11(self.relu(self.conv10(self.max_pool(x))))
        F_cb = self.sig(avg_out + max_out)   # b, c, 1, 1
        F = x * F_cb

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        F_sa = avg_out + max_out
        F_sa = self.sig(self.conv2(F_sa))  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        s_w = self.sig(self.conv11(x_w))
        F_ca = s_h * s_w  # b, c, h, w

        out = F_sa + F_ca

        F = F * out

        return F

class SE_CA_SA(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(SE_CA_SA, self).__init__()
        # CA
        mch = max(8,channel//ratio)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        # CB
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        avg_out = self.conv11(self.relu(self.conv10(self.avg_pool(x))))
        max_out = self.conv11(self.relu(self.conv10(self.max_pool(x))))
        F_cb = self.sig(avg_out + max_out)   # b, c, 1, 1
        F = x * F_cb

        F_sa = torch.mean(x, dim=1, keepdim=True)
        F_sa = self.sig(self.conv2(F_sa))  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        s_w = self.sig(self.conv11(x_w))
        F_ca = s_h * s_w  # b, c, h, w

        out = F_sa + F_ca

        F = F * out

        return F

class SE_CA_SA2(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(SE_CA_SA2, self).__init__()
        # CA
        mch = max(8,channel//ratio)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        # CB
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        avg_out = self.conv11(self.relu(self.conv10(self.avg_pool(x))))
        max_out = self.conv11(self.relu(self.conv10(self.max_pool(x))))
        F_cb = self.sig(avg_out + max_out)   # b, c, 1, 1
        F = x * F_cb

        F_sa = torch.mean(x, dim=1, keepdim=True)
        F_sa = self.conv2(F_sa)  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.conv11(x_h.permute(0, 1, 3, 2))
        s_w = self.conv11(x_w)
        F_ca = s_h * s_w  # b, c, h, w

        out = self.sig(F_sa + F_ca)

        F = F * out

        return F

class EC_CA_SA(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(EC_CA_SA, self).__init__()
        # CA
        mch = max(8, channel//ratio)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        gamma = 2
        b = 1
        ks = int(abs((math.log(channel, 2) + b) / gamma))
        ks = ks if ks % 2 else ks + 1

        padding1 = ks // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv20 = nn.Conv1d(1, 1, kernel_size=ks, padding=padding1, bias=False)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv21 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)
        # self.conv3 = nn.Conv2d(channel, channel, 3, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        F_ch = self.avg(x).view([b, 1, c])
        F_ch = self.conv20(F_ch).view([b, c, 1, 1])  # b,c,1,1

        F_sa = torch.mean(x, dim=1, keepdim=True)
        F_sa = self.conv21(F_sa)  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.conv11(x_h.permute(0, 1, 3, 2))
        s_w = self.conv11(x_w)
        F_ca = s_h * s_w  # b, c, h, w
        F = self.sig(F_ch + F_ca + F_sa)

        out = x * F

        return out

class EC_CA_SA2(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(EC_CA_SA2, self).__init__()
        # CA
        mch = max(8, channel//ratio)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        gamma = 2
        b = 1
        ks = int(abs((math.log(channel, 2) + b) / gamma))
        ks = ks if ks % 2 else ks + 1

        padding1 = ks // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv20 = nn.Conv1d(1, 1, kernel_size=ks, padding=padding1, bias=False)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv21 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)
        # self.conv3 = nn.Conv2d(channel, channel, 3, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        F_ch = self.avg(x).view([b, 1, c])
        F_ch = self.conv20(F_ch).view([b, c, 1, 1])  # b,c,1,1

        F_sa = torch.mean(x, dim=1, keepdim=True)
        F_sa = self.sig(self.conv21(F_sa))  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        s_w = self.sig(self.conv11(x_w))
        F_ca = s_h * s_w  # b, c, h, w
        F = F_ch + F_ca + F_sa

        out = x * F

        return out

class EC_CA_SA3(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(EC_CA_SA3, self).__init__()
        # CA
        mch = max(8, channel//ratio)
        self.conv10 = nn.Conv2d(in_channels=channel, out_channels=mch, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn10 = nn.BatchNorm2d(mch)
        self.conv11 = nn.Conv2d(in_channels=mch, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sig = nn.Sigmoid()

        gamma = 2
        b = 1
        ks = int(abs((math.log(channel, 2) + b) / gamma))
        ks = ks if ks % 2 else ks + 1

        padding1 = ks // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv20 = nn.Conv1d(1, 1, kernel_size=ks, padding=padding1, bias=False)
        # 利用1x1卷积代替全连接

        # SA
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding2 = 3 if kernel_size == 7 else 1
        self.conv21 = nn.Conv2d(1, 1, kernel_size, padding=padding2, bias=False)
        # self.conv3 = nn.Conv2d(channel, channel, 3, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        F_ch = self.avg(x).view([b, 1, c])
        F_ch = self.conv20(F_ch).view([b, c, 1, 1])  # b,c,1,1
        F = x * F_ch

        F_sa = torch.mean(x, dim=1, keepdim=True)
        F_sa = self.sig(self.conv21(F_sa))  # b,1,h,w

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn10(self.conv10(torch.cat((x_h, x_w), 3))))
        x_h, x_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sig(self.conv11(x_h.permute(0, 1, 3, 2)))
        s_w = self.sig(self.conv11(x_w))
        F_ca = s_h * s_w  # b, c, h, w
        F_ca_sa =  F_ca + F_sa

        out = F * F_ca_sa

        return out

class eac(nn.Module):
    def __init__(self, in_planes, gamma=2, b=1):
        super(eac, self).__init__()

        kernel_size = int(abs((math.log(in_planes, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sig(y).view([b, c, 1, 1])
        return x * y
class eac_t(nn.Module):
    def __init__(self, in_planes, gamma=2, b=1):
        super(eac_t, self).__init__()

        kernel_size = int(abs((math.log(in_planes, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.max = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sig(y)
        val1= [item.cpu().detach().numpy() for item in y[0][0]]
        for i in val1:
            f = open(nano_eca,'a')
            a = "%f,"%(i)
            f.write(str(a),)
            f.close
        y = y.view([b, c, 1, 1])
        return x * y


class A2M(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(A2M, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=2 + kernel_size, padding = 1 + padding,  bias=False)
        # self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size+2, padding = padding+1,  bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        # max0 = self.max(x).view([b, 1, c])

        output0 =  self.sig(self.conv1(avg0))
        output0 = self.conv2(output0)
        y0 = self.sig(output0).view([b, c, 1, 1])
        out = x * y0
        output1 = self.conv1(avg0)
        # output1 = self.conv2(output1)
        y1 = self.sig(output1).view([b, c, 1, 1])
        out = out * y1
        return out
class O2S(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(O2S, self).__init__()
        # 计算卷积核大小
        # channel = int((channel*4)/3)
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.max = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d( 1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv1d( 1, 1, kernel_size=2+kernel_size, padding=1+padding,bias=False)
        # self.conv4 = nn.Conv1d( 1, 1, kernel_size=3*kernel_size, padding=6*padding, groups=1, dilation=4,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        output0 = self.conv1(avg0)
        output0 = self.sig(output0)
        output1 = self.conv2(output0)
        y = self.sig(output1).view([b, c, 1, 1])
        out = x * y
        return out

class O2s_sp(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(O2s_sp, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.max = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d( 1, 1, kernel_size=kernel_size+0, padding=padding+0, bias=False)
        self.conv2 = nn.Conv1d( 1, 1, kernel_size=kernel_size+2, padding=padding+1, bias=False)
        self.sig = nn.Sigmoid()
        # self.sm = nn.Softmax()
        self.sp = nn.Softplus()
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        output0 = self.conv1(avg0)
        output0 = self.sig(output0)
        output1 = self.conv2(output0)
        y = 0.85 * self.sp(output1).view([b, c, 1, 1])
        out = x * y
        return out

class O2s_sp_t(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(O2s_sp_t, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.max = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d( 1, 1, kernel_size=kernel_size+0, padding=padding+0, bias=False)
        self.conv2 = nn.Conv1d( 1, 1, kernel_size=kernel_size+2, padding=padding+1, bias=False)
        self.sig = nn.Sigmoid()
        # self.sm = nn.Softmax()
        self.sp = nn.Softplus()
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        output0 = self.conv1(avg0)
        output0 = self.sig(output0)
        val1= [item.cpu().detach().numpy() for item in output0[0][0]]
        for i in val1:
            f = open(nano_sps1410,'a')
            a = "%f,"%(i)
            f.write(str(a),)
            f.close
        output1 = self.conv2(output0)
        output1 = self.conv2(output0)
        y = 0.85 * self.sp(output1)
        val2= [item.cpu().detach().numpy() for item in y[0][0]]
        for i in val2:
            f = open(nano_sps2410,'a')
            a = "%f,"%(i)
            f.write(str(a),)
            f.close

        y = y.view([b, c, 1, 1])
        out = x * y
        return out


class EC_SA(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(EC_SA, self).__init__()
        self.channelattention = eac(channel)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = x * self.spatialattention(x)

        return x

class EC_CA2(nn.Module):
    def __init__(self, channel):
        super(EC_CA2, self).__init__()
        self.channelattention = eac(channel)
        self.spatialattention = CA_Block(channel)

    def forward(self, x):
        x = self.spatialattention(x)
        x = self.channelattention(x)

        return x

class EC_CA3(nn.Module):
    def __init__(self, channel, reduction=16):
        super(EC_CA3, self).__init__()

        self.conv11 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.conv12 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat = self.relu(self.bn(self.conv11(torch.cat((x_h, x_w), 3))))
        x_cat_h, x_cat_w = x_cat.split([h, w], 3)

        x_h = self.conv12(x_cat_h.permute(0, 1, 3, 2))
        x_w = self.conv12(x_cat_w)
        F1 = x_h.expand_as(x) * x_w.expand_as(x)

        F2 = self.avg(x).view([b, 1, c])
        F2 = self.conv(F2).view([b, c, 1, 1])  # b,c,1,1
        F = self.sig(F1 + F2)

        return x * F

class EC_ELA1(nn.Module):
    def __init__(self, channel, ks=7, gamma=2, b=1):
        super(EC_ELA1, self).__init__()

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding1 = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding1, bias=False)

        p = ks // 2
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=ks, padding=p, groups=channel//8, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        F1 = self.avg(x).view([b, 1, c])
        F1= self.conv1(F1).view([b, c, 1, 1])  # b,c,1,1

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_h = self.conv2(x_h).view(b, c, h, 1)
        x_w = self.conv2(x_w).view(b, c, 1, w)
        F2 = x_h * x_w

        F = self.sig(F1 * F2)

        return x * F


class EC_SA33(nn.Module):
    def __init__(self, channel, kernel=7):
        super(EC_SA33, self).__init__()

        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding1 = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding1, bias=False)
        self.sig = nn.Sigmoid()

        padding2 = 3 if kernel== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, kernel, padding=padding2, bias=False)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        b, c, h, w = x.size()

        F1 = self.avg(x).view([b, 1, c])
        F1= self.conv1(F1).view([b, c, 1, 1])  # b,c,1,1

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        F2 = torch.cat([avg_out, max_out], dim=1)
        F2 = self.conv2(F2).expand([b, c, h, w ]) # F2 = self.conv2(F2).expand_as(x), F2 = self.conv2(F2).expand([-1, c, -1, -1 ])
        F2 = self.bn(F2)
        F = self.sig(F1 + F2)

        return x * F

class EC_SA39(nn.Module):
    def __init__(self, channel, kernel=7):
        super(EC_SA39, self).__init__()

        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding1 = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding1, bias=False)
        self.sig = nn.Sigmoid()

        # padding2 = 3 if kernel== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, h, w = x.size()

        F1 = self.avg(x).view([b, 1, c])
        F1= self.conv1(F1).view([b, c, 1, 1])  # b,c,1,1
        F1 = self.sig(F1)
        F = x * F1

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        F2 = torch.cat([avg_out, max_out], dim=1)
        F2 = self.conv2(F2).view([b, h, w ])
        # print(F2.shape)
        F2_h = nn.Conv1d(in_channels=h, out_channels=h, kernel_size=3, padding=1, bias=False).cuda()(F2).view([b, 1, h, w])
        F2_w = nn.Conv1d(in_channels=w, out_channels=w, kernel_size=3, padding=1, bias=False).cuda()(F2.permute(0, 2, 1))
        F2_w = (F2_w.permute(0,2,1)).view([b,1,h,w])
        F2 = F2_h + F2_w
        F2 = self.bn(F2)
        F2 = self.sig(F2)
        F = F * F2

        return  F

class EC_SA41(nn.Module):
    def __init__(self, channel, kernel=7):
        super(EC_SA41, self).__init__()

        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding1 = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding1, bias=False)
        self.sig = nn.Sigmoid()

        # padding2 = 3 if kernel== 7 else 1
        # self.conv2 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, h, w = x.size()

        F1 = self.avg(x).view([b, 1, c])
        F1= self.conv1(F1).view([b, c, 1, 1])  # b,c,1,1
        F1 = self.sig(F1)
        F = x * F1

        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # F2 = torch.cat([avg_out, max_out], dim=1)
        F2 = avg_out.view([b, h, w])
        # print(F2.shape)
        F2_h = nn.Conv1d(in_channels=h, out_channels=h, kernel_size=3, padding=1, bias=False).cuda()(F2).view([b, 1, h, w])
        F2_w = nn.Conv1d(in_channels=w, out_channels=w, kernel_size=3, padding=1, bias=False).cuda()(F2.permute(0, 2, 1))
        F2_w = (F2_w.permute(0,2,1)).view([b,1,h,w])
        F2 = F2_h + F2_w
        F2 = self.bn(F2)
        F2 = self.sig(F2)
        F = F * F2

        return  F

class EC_SA42(nn.Module):
    def __init__(self, channel):
        super(EC_SA42, self).__init__()

        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding1 = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding1, bias=False)
        self.sig = nn.Sigmoid()

        # padding2 = 3 if kernel== 7 else 1
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, h, w = x.size()

        F1 = self.avg(x).view([b, 1, c])
        F1= self.conv1(F1).view([b, c, 1, 1])  # b,c,1,1
        F1 = self.sig(F1)
        F = x * F1

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        F2 = torch.cat([avg_out, max_out], dim=1)
        F2 = self.conv2(F2).view([b, h, w])
        # print(F2.shape)
        F2_h = nn.Conv1d(in_channels=h, out_channels=h, kernel_size=3, padding=1, bias=False).cuda()(F2).view([b, 1, h, w])
        F2_w = nn.Conv1d(in_channels=w, out_channels=w, kernel_size=3, padding=1, bias=False).cuda()(F2.permute(0, 2, 1))
        F2_w = (F2_w.permute(0,2,1)).view([b,1,h,w])
        F2 = F2_h + F2_w
        # F2 = self.bn(F2)
        F2 = self.sig(F2)
        F = F * F2

        return  F

class EC_m_SA(nn.Module):
    def __init__(self, channel, kernel=7):
        super(EC_m_SA, self).__init__()

        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding1 = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size , padding=padding1 , bias=False)
        self.sig = nn.Sigmoid()

        padding2 = 3 if kernel == 7 else 1
        self.conv2 = nn.Conv2d(2, 1, kernel, padding=padding2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.avg(x).view([b, 1, c])
        y = self.conv1(y)
        F1 = self.sig(y).view([b, c, 1, 1])
        F = x * F1

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        F2 = torch.cat([avg_out, max_out], dim=1)
        F2 = self.sig(self.bn(self.conv2(F2)))

        F = x * F

        return F

class EC_SA3(nn.Module):
    def __init__(self, channel, kernel=7):
        super(EC_SA3, self).__init__()

        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding1 = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size , padding=padding1 , bias=False)
        self.sig = nn.Sigmoid()

        padding2 = 3 if kernel == 7 else 1
        self.conv2 = nn.Conv2d(1, 1, kernel, padding=padding2, bias=False)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.avg(x).view([b, 1, c])
        F1 = self.conv1(y).view([b, c, 1, 1])

        F2 = torch.mean(x, dim=1, keepdim=True)
        F2 = self.bn(self.conv2(F2).expand([b,c,h,w]))

        F = self.sig(F1 + F2)

        return x * F

class SKnet(nn.Module):
    def __init__(self, features, WH=1, M=2, G=32, r=16, stride=1, L=32):
        super(SKnet, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

# adaptive
class O2Ss(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(O2Ss, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.max = nn.AdaptiveMaxPool2d(1)
        # self.alpha = alpha
        self.conv1 = nn.Conv1d( 1, 1, kernel_size=kernel_size+0, padding=padding+0, bias=False)
        self.conv2 = nn.Conv1d( 1, 1, kernel_size=kernel_size+2, padding=padding+1, bias=False)
        # self.conv4 = nn.Conv1d( 1, 1, kernel_size=3*kernel_size, padding=6*padding, groups=1, dilation=4,bias=False)
        self.sig = nn.Sigmoid()
        # self.sm = nn.Softmax()
        # self.sp = nn.Softplus()
        self.ss30 = ms30()
        self.ss40 = ms40()
        self.ss20 = ms20()
        self.ss15 = ms15()
        self.ss075 = ms075()
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        input0 = avg0
        output0 = self.conv1(input0)

        if c >= 512:
            output0 = self.sig(output0)
            output1 = self.conv2(output0)
            y = self.sig(output1).view([b, c, 1, 1])
        elif 512> c >= 384:
            output0 = self.sig(output0)
            output1 = self.conv2(output0)
            y = self.ss15(output1).view([b, c, 1, 1])
        elif 384> c >= 256:
            output0 = self.sig(output0)
            output1 = self.conv2(output0)
            y = self.ss20(output1).view([b, c, 1, 1])
        elif 256> c >= 128:
            output0 = self.sig(output0)
            output1 = self.conv2(output0)
            y = self.ss30(output1).view([b, c, 1, 1])
        else:
            output0 = self.sig(output0)
            output1 = self.conv2(output0)
            y = self.ss40(output1).view([b, c, 1, 1])

        out = x * y
        return out


class O2Sr(nn.Module):
    def __init__(self, channel, gamma=1.3, b=2):
        super(O2Sr, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(2*channel+100) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.max = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d( 1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv1d( 1, 1, kernel_size=2+kernel_size, padding=1+padding, bias=False)
        # self.conv4 = nn.Conv1d( 1, 1, kernel_size=3*kernel_size, padding=6*padding, groups=1, dilation=4,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        input0 = avg0
        output0 = self.conv1(input0)
        output0 = self.sig(output0)
        output1 = self.conv2(output0)
        y = self.sig(output1).view([b, c, 1, 1])
        out = x * y
        return out

#
class O2Sr_sp(nn.Module):
    # def __init__(self, channel, gamma=2, b=1):
    #     super(O2Sr_sp, self).__init__()
    #     # 计算卷积核大小
    #     kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    #     kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    def __init__(self, channel, gamma=1.6, b=1.2):
        super(O2Sr_sp, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel+100) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.max = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d( 1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv1d( 1, 1, kernel_size=2+kernel_size, padding=1+padding, bias=False)
        # self.conv4 = nn.Conv1d( 1, 1, kernel_size=3*kernel_size, padding=6*padding, groups=1, dilation=4,bias=False)
        self.sig = nn.Sigmoid()
        self.sp = nn.Softplus()
        # self.a = 1
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        output0 = self.conv1(avg0)
        output0 = self.sig(output0)
        output1 = self.conv2(output0)
        if c<=64:
            y = self.sp(output1).view([b, c, 1, 1])
        else:
            y = self.sig(output1).view([b, c, 1, 1])
        # y = self.sig(output1).view([b, c, 1, 1])
        out = x * y
        return out
# test weights
class O2S_t(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(O2S_t, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d( 1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv1d( 1, 1, kernel_size=2+kernel_size, padding=1+padding,bias=False)
        self.sig = nn.Sigmoid()
        # self.sp = nn.Softplus()
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        avg0 = self.avg(x).view([b, 1, c])
        output0 = self.conv1(avg0)
        output0 = self.sig(output0)

        val1= [item.cpu().detach().numpy() for item in output0[0][0]]
        for i in val1:
            f = open(nano_o2s1,'a')
            a = "%f,"%(i)
            f.write(str(a),)
            f.close
        output1 = self.conv2(output0)

        y = self.sig(output1)
        val2 = [item.cpu().detach().numpy() for item in y[0][0]]
        for i in val2:
            f = open(nano_o2s2,'a')
            a = "%f,"%(i)
            f.write(str(a),)
            f.close
        y = y.view([b, c, 1, 1])
        out = x * y
        return out