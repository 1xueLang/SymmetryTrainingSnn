import torch
import torch.nn as nn
import snetx.snn.algorithm as snnalgo
import snetx.cuend.neuron as neuron

class SymmConvSequential(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, tau=2., scale=0.5, symm_training=True, symm_connect=False, pooling=None):
        super().__init__()
        annconv = [nn.ReLU(),]
        snnconv = []
        if pooling != None:
            annconv += [pooling]
            snnconv += [pooling]
        annconv += [
            nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
        ]
        self.annconv = nn.Sequential(*annconv)
        snnconv += [
            nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
        ]
        self.snnconv = nn.Sequential(
            neuron.LIF(tau=tau),
            snnalgo.Tosnn(nn.Sequential(*snnconv))
        )
        self.scale = scale
        self.symm_training = symm_training
        self.symm_connect = symm_connect

    def forward(self, x):
        snnin, annin = x
        snnh = self.snnconv(snnin)
        annh = self.annconv(annin)
        if self.symm_training:
            annout = annh * (1. - self.scale) + snnh.mean(dim=1) * self.scale
        else:
            annout = annh
        if self.symm_connect:
            snnout = snnh * (1. - self.scale) + annh.unsqueeze(dim=1) * self.scale
        else:
            snnout = snnh
        return (snnout, annout)

class SymmConvEncoder(nn.Module):
    def __init__(self, conv_config, scale=0.5, symm_training=True, symm_connect=False, **kwargs):
        super().__init__()
        out_planes = conv_config['out_channels']
        self.ann = nn.Sequential(
            nn.Conv2d(**conv_config), nn.BatchNorm2d(out_planes)
        )
        self.snn = snnalgo.Tosnn(
            nn.Sequential(nn.Conv2d(**conv_config), nn.BatchNorm2d(out_planes))
        )
        self.scale = scale
        self.symm_training = symm_training
        self.symm_connect = symm_connect

    def forward(self, x):
        snnin, annin = x
        snnh = self.snn(snnin)
        annh = self.ann(annin)
        if self.symm_training:
            annout = annh * (1. - self.scale) + snnh.mean(dim=1) * self.scale
        else:
            annout = annh
        if self.symm_connect:
            snnout = snnh * (1. - self.scale) + annh.unsqueeze(dim=1) * self.scale
        else:
            snnout = snnh
        return (snnout, annout)


class SymmLinearSequential(nn.Module):
    def __init__(self, in_planes, out_planes, tau=2.0, scale=0.5, symm_training=True, symm_connect=False, **kwargs):
        super().__init__()
        self.annlinear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_planes, out_planes),
            # nn.BatchNorm1d(out_planes),
        )
        self.snnlinear = nn.Sequential(
            neuron.LIF(tau=tau),
            snnalgo.Tosnn(
                nn.Linear(in_planes, out_planes)
            )
        )
        self.scale = scale
        self.symm_training = symm_training
        self.symm_connect = symm_connect

    def forward(self, x):
        snnin, annin = x
        snnh = self.snnlinear(snnin)
        annh = self.annlinear(annin)
        if self.symm_training:
            annout = annh * (1. - self.scale) + snnh.mean(dim=1) * self.scale
        else:
            annout = annh
        if self.symm_connect:
            snnout = snnh * (1. - self.scale) + annh.unsqueeze(dim=1) * self.scale
        else:
            snnout = snnh
        return (snnout, annout)

class SymmForward(nn.Module):
    def __init__(self, snn, ann, scale=0.5, symm_training=True, symm_connect=False, **kwargs):
        super().__init__()
        self.ann = ann
        self.snn = snn
        self.scale = scale
        self.symm_training = symm_training
        self.symm_connect = symm_connect

    def forward(self, x):
        sin, ain = x
        sh = self.snn(sin)
        ah = self.ann(ain)
        if self.symm_connect:
            sout = (1. - self.scale) * sh + self.scale * ah.unsqueeze(dim=1)
        else:
            sout = sh
        if self.symm_training:
            aout = (1. - self.scale) * ah + self.scale * sh.mean(dim=1)
        else:
            aout = ah
        return (sout, aout)