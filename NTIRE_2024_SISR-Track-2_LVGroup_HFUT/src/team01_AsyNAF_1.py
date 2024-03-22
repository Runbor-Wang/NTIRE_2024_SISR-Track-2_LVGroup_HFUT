# ------------------------------------------------------------------------
# Thanks to the NAFNet baseline: https://github.com/megvii-research/NAFNet
# Thanks to the TLC: https://github.com/megvii-research/TLC
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnPixelShuffle(nn.Module):
    # Rearranges elements in a tensor of shape (N, C, H*r, W*r) to a tensor of shape (N, C*(r^2), H, W) where r is an downscale factor.
    def __init__(self, downscale_factor):
        super(UnPixelShuffle, self).__init__()
            
        self.downscale_factor = downscale_factor
            
    def forward(self, x):
        N, C, H, W = x.size()
            
        device = x.device

        chunk_list = []
            
        for i in range(self.downscale_factor):
            for j in range(self.downscale_factor):
                chunk_list.append(x[..., i:H:self.downscale_factor, j:W:self.downscale_factor])
                    
        x = torch.cat(chunk_list, dim=1)
            
        index = []
        for i in range(C):
            for j in range(self.downscale_factor**2):
                index.append(i+j*C)
        x = torch.gather(x, dim=1, index=torch.tensor(index).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(N,1,H//self.downscale_factor,W//self.downscale_factor).to(device))
            
        return x

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


'''
ref. 
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''
class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


# class LayerNormFunction(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, x, weight, bias, eps):
#         ctx.eps = eps
#         N, C, H, W = x.size()
#         mu = x.mean(1, keepdim=True)
#         var = (x - mu).pow(2).mean(1, keepdim=True)
#         y = (x - mu) / (var + eps).sqrt()
#         ctx.save_for_backward(y, var, weight)
#         y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         eps = ctx.eps

#         N, C, H, W = grad_output.size()
#         y, var, weight = ctx.saved_variables
#         g = grad_output * weight.view(1, C, 1, 1)
#         mean_g = g.mean(dim=1, keepdim=True)

#         mean_gy = (g * y).mean(dim=1, keepdim=True)
#         gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
#         return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
#             dim=0), None

# class LayerNorm2d(nn.Module):

#     def __init__(self, channels, eps=1e-6):
#         super(LayerNorm2d, self).__init__()
#         self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
#         self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
#         self.eps = eps

#     def forward(self, x):
#         return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# class SimpleGate(nn.Module):
#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         return x1 * x2

# class NAFBlockOriginal(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
#                                bias=True)
#         self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )

#         # SimpleGate
#         self.sg = SimpleGate()

#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)

#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

#     def forward(self, inp):
#         x = inp

#         x = self.norm1(x)

#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.sg(x)
#         x = x * self.sca(x)
#         x = self.conv3(x)

#         x = self.dropout1(x)

#         y = inp + x * self.beta

#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)

#         x = self.dropout2(x)

#         return y + x * self.gamma

# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
#                                bias=True)
#         self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )

#         # SimpleGate
#         self.sg = SimpleGate()

#         self.norm1 = LayerNorm2d(c)

#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

#     def forward(self, inp):
#         x = inp

#         x = self.norm1(x)

#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.sg(x)
#         x = x * self.sca(x)
#         x = self.conv3(x)

#         x = self.dropout1(x)

#         y = inp + x * self.beta

#         return y

# class AsyNAF(nn.Module):

#     def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
#         super().__init__()

#         self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
#                               bias=True)

#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()
#         self.middle_blks = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()

#         chan = width

#         for num in enc_blk_nums:
#             self.encoders.append(
#                 nn.Sequential(
#                     *[NAFBlock(chan) for _ in range(num)]
#                 )
#             )
#             self.downs.append(
#                 nn.Sequential(
#                     UnPixelShuffle(2)
#                 )
#             )
#             chan = chan * 4

#         self.middle_blks = \
#             nn.Sequential(
#                 *[NAFBlock(chan) for _ in range(middle_blk_num)]
#             )

#         for num in dec_blk_nums:
#             self.ups.append(
#                 nn.Sequential(
#                     nn.Conv2d(chan, chan, 1, bias=False),
#                     nn.PixelShuffle(2)
#                 )
#             )
#             chan = chan // 4
#             self.decoders.append(
#                 nn.Sequential(
#                     *[NAFBlock(chan) for _ in range(num)]
#                 )
#             )

#         self.up_final = nn.Sequential(
#                     nn.Conv2d(chan, img_channel*16, 1, bias=False),
#                     nn.PixelShuffle(4)
#                 )

#         self.padder_size = 2 ** len(self.encoders)

#     def forward(self, inp):
#         B, C, H, W = inp.shape
#         inp = self.check_image_size(inp)

#         pre = self.intro(inp)

#         x = pre

#         encs = []

#         for encoder, down in zip(self.encoders, self.downs):
#             x = encoder(x)
#             encs.append(x)
#             x = down(x)

#         x = self.middle_blks(x)

#         for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
#             x = up(x)
#             x = x + enc_skip
#             x = decoder(x)
        
#         x = x + pre

#         x = self.up_final(x)

#         return x[:, :, :H*4, :W*4]

#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
#         mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
#         return x

# # class AsyNAF(nn.Module):

# #     def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
# #         super().__init__()

# #         self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
# #                               bias=True)
# #         self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
# #                               bias=True)

# #         self.encoders = nn.ModuleList()
# #         self.decoders = nn.ModuleList()
# #         self.middle_blks = nn.ModuleList()
# #         self.ups = nn.ModuleList()

# #         chan = width

# #         for num in enc_blk_nums:
# #             self.encoders.append(
# #                 nn.Sequential(
# #                     *[NAFBlock(chan) for _ in range(num)]
# #                 )
# #             )

# #         self.middle_blks = \
# #             nn.Sequential(
# #                 *[NAFBlock(chan) for _ in range(middle_blk_num)]
# #             )

# #         for num in dec_blk_nums:
# #             self.ups.append(
# #                 nn.Sequential(
# #                     nn.Conv2d(chan, chan * 16, 1, bias=False),
# #                     nn.PixelShuffle(4)
# #                 )
# #             )
# #             self.decoders.append(
# #                 nn.Sequential(
# #                     *[NAFBlock(chan) for _ in range(num)]
# #                 )
# #             )

# #     def forward(self, inp):
        
# #         x = self.intro(inp)

# #         encs = []

# #         s_factor = 4
# #         for encoder in self.encoders:
# #             x = encoder(x)
# #             encs.append(F.interpolate(x, scale_factor=s_factor, mode='area'))
# #             s_factor /= 4

# #         x = self.middle_blks(x)
        
# #         x = x + encs[-1]

# #         for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[:-1][::-1]):
# #             x = up(x)
# #             x = x + enc_skip
# #             x = decoder(x)

# #         x = self.ending(x)
# #         x = x + F.interpolate(inp, scale_factor=4, mode='area')

# #         return x
    
# class AsyNAFLocal(Local_Base, AsyNAF):
#     def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         AsyNAF.__init__(self, *args, **kwargs)

#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))

#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# if __name__ == '__main__':
#     def print_para_num(model):
    
#         '''
#         function: print the number of total parameters and trainable parameters
#         '''

#         total_params = sum(p.numel() for p in model.parameters())
#         total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print('total parameters: %d' % total_params)
#         print('trainable parameters: %d' % total_trainable_params)
    
#     img_channel = 3
#     width = 32

#     # enc_blks = [2, 2, 4, 8]
#     # middle_blk_num = 12
#     # dec_blks = [2, 2, 2, 2]

#     enc_blks = [1, 1, 1, 28]
#     middle_blk_num = 1
#     dec_blks = [1, 1, 1, 1]
    
#     net = AsyNAF(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
#                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

#     print_para_num(net)
    
#     inp_shape = (3, 256, 256)

#     from ptflops import get_model_complexity_info

#     macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

#     # params = float(params[:-3])
#     # macs = float(macs[:-4])

#     print(macs, params)

# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True,
               depth_wise=False):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
               int((kernel_size[1] - 1) / 2))
    if not depth_wise:
        return nn.Conv2d(in_channels,
                         out_channels,
                         kernel_size,
                         padding=padding,
                         bias=bias)
    else:
        return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=in_channels),
                    nn.LeakyReLU(0.05, inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
                )


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3, depth_wise=True)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out


class RLFN_Prune(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in NTIRE 2022 Efficient SR Challenge
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=46,
                 mid_channels=48,
                 upscale=4):
        super(RLFN_Prune, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = RLFB(feature_channels, mid_channels)
        self.block_2 = RLFB(feature_channels, mid_channels)
        self.block_3 = RLFB(feature_channels, mid_channels)
        self.block_4 = RLFB(feature_channels, mid_channels)

        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)

        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)
        output = output + F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return output

class RLFN_PruneLocal(Local_Base, RLFN_Prune):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        RLFN_Prune.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)