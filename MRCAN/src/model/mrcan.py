
from model import common
import torch
import torch.nn as nn


def make_model(args, parent=False):
    return MRCAN(args)

## Multi-Scale Convolutional Layer
class MSCLayer(nn.Module):
    def __init__(self):
        super(MSCLayer, self).__init__()

        channel = 64

        self.conv_3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv_5 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1, padding=2,
                                bias=True)
        self.confusion = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                   bias=True)

    def forward(self, x):
        output_3 = self.conv_3(x)
        output_5 = self.conv_5(x)
        output = torch.cat([output_3, output_5], 1)
        output = self.confusion(output)  # 使用1x1卷积调整通道数
        return output

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Multiscale Residual Channel Space Attention Block (MRCSAB)
class MRCAB(nn.Module):
    def __init__(self, conv, n_feat, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MRCAB, self).__init__()
        modules_body1 = []
        for i in range(2):
            modules_body1.append(conv(n_feat, n_feat, kernel_size=3, bias=bias))
            if bn: modules_body1.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body1.append(act)
        modules_body2 = []
        for i in range(2):
            modules_body2.append(conv(n_feat, n_feat, kernel_size=5, bias=bias))
            if bn: modules_body2.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body2.append(act)
        self.body1 = nn.Sequential(*modules_body1)
        self.body2 = nn.Sequential(*modules_body2)
        self.conv1x1 = conv(2 * n_feat, n_feat, kernel_size=1, bias=bias)
        if bn:
            self.bn1x1 = nn.BatchNorm2d(n_feat)
        self.res_scale = res_scale
        self.ca = CALayer(n_feat, reduction)

    def forward(self, x):
        x1 = self.body1(x)
        x2 = self.body2(x)
        res = torch.cat([x1, x2], 1)
        # res = self.body(x).mul(self.res_scale)
        res = self.conv1x1(res)
        if hasattr(self, 'bn1x1'):
            res = self.bn1x1(res)
        res = self.ca(res)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            MRCAB(
                conv, n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Multiscale Residual Channel Space Attention Network (MRCSAN)
class MRCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MRCAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)

        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.msclayer = MSCLayer()
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.msclayer(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
