import torch
import torch.nn as nn
from . import dhtcu_block as B
def make_model(args, parent=False):
    model = HUTCN()
    return model


class Cascade(nn.Module):
    def __init__(self, ):
        super(Cascade, self).__init__()
        self.conv1 = B.conv_layer(50, 50, kernel_size=1)
        self.conv3 = B.conv_layer(50, 50, kernel_size=3)
        self.conv5 = B.conv_layer(50, 50, kernel_size=5)
        self.c = B.conv_block(50 * 4, 50, kernel_size=1, act_type='lrelu')

    def forward(self, x):
        conv5 = self.conv5(x)
        extra = x+conv5
        conv3 = self.conv3(extra)
        extra = x + conv3
        conv1 = self.conv1(extra)
        cat = torch.cat([conv5, conv3, conv1, x], dim=1)
        input = self.c(cat)
        return input


class HUTCN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=3):
        super(HUTCN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.rc = self.remaining_channels = in_nc
        self.c1_r = B.conv_layer(in_nc, self.rc, 3)

        self.B1 = B.P_HTCB(in_channels=nf)
        self.B2 = B.P_HTCB(in_channels=nf)
        self.B3 = B.P_HTCB(in_channels=nf)
        self.B4 = B.P_HTCB(in_channels=nf)
        self.B5 = B.P_HTCB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)      
        
        out_B4 = self.B4(out_B3)
        out_B4 = self.LR_conv(out_B3) + out_B2
        
        out_B5 = self.B5(out_B4)
        out_B5 = self.LR_conv(out_B4) + out_B1
        
        out_lr = self.LR_conv(out_B5) + out_fea
        
        output = self.upsampler(out_lr)

        return output
    
    
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
