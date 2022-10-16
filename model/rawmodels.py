# Based on:
# https://github.com/rosinality/swapping-autoencoder-pytorch

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from .components import *

from .stylegan2.model import StyledConv, EqualLinear, EqualConv2d

# Swapping AE
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels = 3,
        entry_channels = 32,
        structure_channels=8,
        texture_channels=2048,
        blur_kernel=(1, 3, 3, 1),
        downsample = (True, True, True, True)
    ):
        super().__init__()

        # Core Element
        core = [ConvLayer(in_channels, entry_channels, 1)]
        
        downsample = [bool(el) for el in downsample]
        
        n_res = len(downsample)
        for i in range(n_res):
            core.append(ResBlock(entry_channels*(2**i),
                                 entry_channels*(2**(i+1)),
                                 downsample=downsample[i],
                                 padding="reflect"))
        
        ch = entry_channels*(2**n_res)
        
        self.core = nn.Sequential(*core)

        # Structure Processing
        self.structure = nn.Sequential(
            ConvLayer(ch, ch, 1), # downsample=True),  ##### CAUTION
            ConvLayer(ch, structure_channels, 1) ############################
        )

        # Texture Processing
        self.texture = nn.Sequential(
            ConvLayer(ch, ch * 2, 3, downsample=True, padding="valid"),
            ConvLayer(ch * 2, ch * 4, 3, downsample=True, padding="valid"),
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(ch * 4, texture_channels, 1),
        )

    def forward(self, input):
        out = self.core(input)

        structure = self.structure(out)
        texture = torch.flatten(self.texture(out), 1)

        return structure, texture

    
class Generator(nn.Module):
    def __init__(
        self,
        entry_channels = 32,
        structure_channels=8,
        texture_channels=2048,
        blur_kernel=(1, 3, 3, 1),
        sigmoid_output = False,
        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4),
        upsample = (False, False, False, False, True, True, True, True)
    ):
        super().__init__()

        ch_multiplier = [int(el) for el in ch_multiplier]
        upsample = [bool(el) for el in upsample] 

        self.layers = nn.ModuleList()
        in_ch = structure_channels
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, entry_channels * ch_mul, texture_channels, up, blur_kernel
                )
            )
            in_ch = entry_channels * ch_mul

        self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)
        self.sigmoid_output = sigmoid_output

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, texture, noise)

        out = self.to_rgb(out)
        
        if self.sigmoid_output:
            return torch.sigmoid(out)
        else:
            return out
    
    
# Discrimination Models
class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, arch_scale = 1, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size
            
        self.ratio = int(self.size[1]/self.size[0])
        channels = {
            4: int(512*arch_scale),
            8: int(512*arch_scale),
            16: int(512*arch_scale),
            32: int(512*arch_scale),
            64: int(256 * channel_multiplier*arch_scale),
            128: int(128 * channel_multiplier*arch_scale),
            256: int(64 * channel_multiplier*arch_scale),
            512: int(32 * channel_multiplier*arch_scale),
            1024: int(16 * channel_multiplier*arch_scale)
        }

        convs = [ConvLayer(3, channels[self.size[0]], 1)]

        log_size = int(math.log(self.size[0], 2))

        in_channel = channels[self.size[0]]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4 * self.ratio, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        #print(out.shape)
        out = self.final_linear(out)

        return out
    
# class PatchDiscriminator(nn.Module):
#     def __init__(self,
#                  channel,
#                  ch_multiplier = (2, 4, 8, 12, 12, 24),
#                  downsample = (True, True, True, True, True, False),
#                  size = 256):
#         super().__init__()
        
#         self.size = size
#         encoder = [ConvLayer(3, channel, 1)]
        
#         self.ch_multiplier = ch_multiplier
#         self.downsample = downsample
        
#          #True, False)
#         in_ch = channel
#         for ch_mul, down in zip(self.ch_multiplier, self.downsample):
#             encoder.append(ResBlock(in_ch, channel * ch_mul, down))
#             in_ch = channel * ch_mul

#         if self.size > 511:
#             k_size = 3
#             feat_size = 2 * 2     
#         else:
#             k_size = 2
#             feat_size = 1 * 1

#         encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))
#         self.encoder = nn.Sequential(*encoder)

#         self.linear = nn.Sequential(
#             EqualLinear(
#                 channel * 12 * 2 * feat_size, channel * 32, activation="fused_lrelu"
#             ),
#             EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
#             EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
#             EqualLinear(channel * 16, 1),
#         )
        
#     def forward(self, input, reference=None, ref_batch=None, ref_input=None):
#         out_input = self.encoder(input)

#         if ref_input is None:
#             ref_input = self.encoder(reference)
#             _, channel, height, width = ref_input.shape
#             ref_input = ref_input.view(-1, ref_batch, channel, height, width)
#             ref_input = ref_input.mean(1)

#         out = torch.cat((out_input, ref_input), 1)
#         out = torch.flatten(out, 1)

#         out = self.linear(out)

#         return out, ref_input

class StructureDiscriminator(nn.Module):
    # Novel Structure Discriminator
    
    def __init__(self,
                 gradient_norm_input = False,
                 merge_gradient_dimensions = True,
                 image_channels = 3,
                 structure_channels = 8,
                 ratio = 8,
                 channels = (16, 32, 64, 128, 256, 256),
                 downsample = (True, True, True, True, False),
                 classifier_scale = 32,
                 structure_size = 32
                ):
        
        super().__init__()
        
        self.gradient_norm_input = gradient_norm_input
        self.merge_gradient_dimensions = merge_gradient_dimensions
        if not self.gradient_norm_input:
            self.image_channels = image_channels
            self.structure_channels = structure_channels
        else:
            self.image_channels = 1 if self.merge_gradient_dimensions else 2
            self.structure_channels = 1 if self.merge_gradient_dimensions else 2
        self.channels = channels
        self.downsample = downsample
        
        # downsample from image 2 structure
        self.pooling = torch.nn.AvgPool2d(kernel_size = ratio,
                                          stride=ratio,
                                          padding=0)
        
        self.image_entry = ConvLayer(self.image_channels, self.channels[0], 1)
        self.structure_entry = ConvLayer(self.structure_channels, self.channels[0], 1)
        
        #encoder = [ConvLayer(3, channel, 1)]
        encoder = []        

        in_ch = channels[0]
        for out_ch, down in zip(self.channels[1:], self.downsample):
            encoder.append(ResBlock(in_ch, out_ch, down))
            in_ch = out_ch

        down_steps = len(np.nonzero(downsample)[0])
        curr_size = int(structure_size*0.5**(down_steps+1))
        feat_size = curr_size**2
        
        k_size = 2 if curr_size < 2 else 3

        encoder.append(ConvLayer(in_ch, classifier_scale * 12, k_size, padding="valid"))
        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(
                 classifier_scale * 12 * 2 * feat_size, classifier_scale * 32, activation="fused_lrelu"
            ),
            EqualLinear(classifier_scale * 32, classifier_scale * 32, activation="fused_lrelu"),
            EqualLinear(classifier_scale * 32, classifier_scale * 16, activation="fused_lrelu"),
            EqualLinear(classifier_scale * 16, 1),
        )       

#     def forward(self, input, reference = None, ref_batch = None, ref_input = None):
#         out_input = self.encoder(input)

#         if ref_input is None:
#             ref_input = self.encoder(reference)
#             _, channel, height, width = ref_input.shape
#             ref_input = ref_input.view(-1, ref_batch, channel, height, width)
#             ref_input = ref_input.mean(1)
            
#         out = torch.cat((out_input, ref_input), 1)
#         out = torch.flatten(out, 1)

#         out = self.linear(out)

#         return out, ref_input

    def gradient_norm(self, input):
        x_grad = input[0,:,1:,:] - input[0,:,:-1,:]
        y_grad = input[0,:,:,1:] - input[0,:,:,:-1]

        # l2 norm
        x_grad_norm = torch.nn.functional.pad(torch.norm(x_grad, 2, dim = 0), pad = (0,0,0,1))
        y_grad_norm = torch.nn.functional.pad(torch.norm(y_grad, 2, dim = 0), pad = (0,1,0,0))
        
        x_grad_norm = (x_grad_norm/x_grad_norm.var()).view(1,1,*x_grad_norm.shape)
        y_grad_norm = (y_grad_norm/y_grad_norm.var()).view(1,1,*y_grad_norm.shape)
            
        if self.merge_gradient_dimensions:
            return x_grad_norm + y_grad_norm
        else:
            return torch.cat([x_grad_norm, y_grad_norm], 1)
    
    def forward(self, input, structure): 
        if self.gradient_norm_input:
            img_rep = self.gradient_norm(self.pooling(input))
            str_rep = self.gradient_norm(structure)
        else:
            img_rep = self.pooling(input)
            str_rep = structure

        out_input = self.encoder(self.image_entry(img_rep))
        out_struct = self.encoder(self.structure_entry(str_rep))
            
        out = torch.cat((out_input, out_struct), 1)
        out = torch.flatten(out, 1)

        out = self.linear(out)

        return out, out_struct

class PatchDiscriminator(nn.Module):
    # Flexible Patch Discriminator
    # Default Settings yield original implementation from the Swapping AE
    
    def __init__(self,
                 size = 256,
                 channel = 32,
                 ch_multiplier = (2, 4, 8, 12, 12, 24),
                 downsample = (True, True, True, True, True, False),
                 patch_size = 64
                ):
        
        super().__init__()
        
        encoder = [ConvLayer(3, channel, 1)]
        
        self.ch_multiplier = ch_multiplier
        self.downsample = downsample

        in_ch = channel
        for ch_mul, down in zip(self.ch_multiplier, self.downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        down_steps = len(np.nonzero(downsample)[0])
        curr_size = int(patch_size*0.5**(down_steps+1))
        feat_size = curr_size**2
        
        k_size = 2 if curr_size < 2 else 3

        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))
        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(
                 channel * 12 * 2 * feat_size, channel * 32, activation="fused_lrelu"
            ),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )       

    def forward(self, input, reference = None, ref_batch = None, ref_input = None):
        out_input = self.encoder(input)

        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, channel, height, width)
            ref_input = ref_input.mean(1)
            
        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)

        out = self.linear(out)

        return out, ref_input
    
class TextureComparator(nn.Module):
    def __init__(
        self,
        texture_channels = 64,
        channel = 1024
    ):
        super().__init__()
        
        self.comparator = nn.Sequential(
            EqualLinear(
                 2 * texture_channels, 2 * channel, activation="fused_lrelu"
            ),
            EqualLinear(2 * channel, 2 * channel, activation="fused_lrelu"),
            EqualLinear(2 * channel, 1 * channel, activation="fused_lrelu"),
            EqualLinear(1 * channel, 1),
        )       
        
    def forward(self, input, reference):
            
        out = torch.cat((input, reference), 1)
        out = torch.flatten(out, 1)

        out = self.comparator(out)

        return out           
            
# based on https://github.com/tamarott/SinGAN
    
# class ConvBlock(nn.Sequential):
#     def __init__(self, in_channel, out_channel, ker_size, padd, stride):
#         super(ConvBlock,self).__init__()
#         self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
#         self.add_module('norm',),
#         self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

class EveryPatchDiscriminator(nn.Module):
    def __init__(self,
                 channel = 32,
                 kernel_size = 3,
                 num_layer = 5,
                 feat_size = 64,
                 reference_count = -1,
                 **kwargs):
        
        super().__init__()
        
        self.reference_count = reference_count
        
        # number of feature channels
        nfc = channel
        min_nfc = channel
        self.kernel_size = kernel_size
        stride = 1
        padding = 1
                
        self.num_layer = num_layer
        self.feat_size = feat_size
        
        N = nfc
        
        #self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
#         head = nn.Sequential(nn.Conv2d(3, N, kernel_size = kernel_size, stride=stride, padding=padding),
#                                   #nn.BatchNorm2d(N),
#                                   nn.LeakyReLU(0.2, inplace=True)
#                                   )
        head = nn.Sequential(nn.Conv2d(3, N, kernel_size = self.kernel_size, stride=stride,
                                       padding=int(0.5*(self.kernel_size-1))),
                                  #nn.BatchNorm2d(N),
                                  nn.LeakyReLU(0.2, inplace=True)
                                  )
        
        body = nn.Sequential()
        for i in range(self.num_layer-2):
            N = int(nfc/pow(2,(i+1)))
            #block = ConvBlock(,, opt.ker_size, opt.padd_size, 1)            
            block = nn.Sequential(nn.Conv2d(max(2*N, min_nfc),
                                            max(N, min_nfc),
                                            kernel_size = self.kernel_size,
                                            stride=stride,
                                            padding=int(0.5*(self.kernel_size-1))),
                                  #nn.BatchNorm2d(max(N, min_nfc)),
                                  nn.LeakyReLU(0.2, inplace=True)
                                  )
            body.add_module('block%d'%(i+1),block)
            
        tail = nn.Conv2d(max(N,min_nfc),
                         self.feat_size,
                         kernel_size= self.kernel_size,
                         stride=1,
                         padding=int(0.5*(self.kernel_size-1))
                        )
        
        
        self.encoder = nn.Sequential(head, body, tail)   
        
        # Comparator
        self.comparator = nn.Sequential(
            ConvLayer(2 * self.feat_size, 4*self.feat_size, kernel_size = 1),
            ConvLayer(4*self.feat_size, 4*self.feat_size, kernel_size = 1),
            ConvLayer(4*self.feat_size, 2*self.feat_size, kernel_size = 1),
            ConvLayer(2*self.feat_size, 1, kernel_size = 1)
        )
        
        

    def forward(self, input, reference=None, ref_batch=None, ref_input=None, reduce = True, reference_count = -1):
        """
        reference_count - if -1 encoding from all reference patches is used, otherwise the mean of randomly selected reference_count patches is used
        """
        out_input = self.encoder(input)

        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            
            if reference_count != -1 or self.reference_count != -1:
                # shuffle
                ref_input = ref_input[..., torch.randperm(ref_input.shape[-2]),:]
                ref_input = ref_input[..., torch.randperm(ref_input.shape[-1])]             
                # keep 1st dimension, keep 2nd (feature dim), get 1st row, and N first cols (both rows and cols shuffled)
                ref_input = ref_input[:,:,0,:int(reference_count)].mean((-1), keepdim = True).unsqueeze(-1).expand(-1, -1, height, width)                
            else:
                # mean for all patches
                ref_input = ref_input.mean((-1, -2), keepdim = True).expand(-1, -1, height, width)

        out = torch.cat((out_input, ref_input), 1)

        out = self.comparator(out)
        
        if reduce:
            out = torch.flatten(out)
            
        return out, ref_input