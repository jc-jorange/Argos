import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.Pad2D
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.Pad2D
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2D
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        # 定义一系列的高斯核
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt) # 归一化操作，保证特征经过blur后信息总量不变
        # 非grad操作的参数利用buffer存储
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            # 利用固定参数的conv2d+stride实现blurpool
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])