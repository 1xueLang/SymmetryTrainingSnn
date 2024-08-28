import torch
import torch.nn as nn
import snetx.snn.algorithm as snnalgo
import snetx.cuend.neuron as neuron
import symm

class VGG11(nn.Module):
    def __init__(self, symm_config, img_channels, tau=2.0, num_classes=10, dropout=0.5):
        super().__init__()
        self.encod = symm.SymmConvEncoder(
            {'in_channels': img_channels, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}, **symm_config
        )
        self.body = nn.Sequential(
            symm.SymmConvSequential(64, 128, pooling=nn.AvgPool2d(2, 2), **symm_config),
            symm.SymmConvSequential(128, 256, pooling=nn.AvgPool2d(2, 2), **symm_config),
            symm.SymmConvSequential(256, 256, **symm_config),
            symm.SymmConvSequential(256, 512, pooling=nn.AvgPool2d(2, 2), **symm_config),
            symm.SymmConvSequential(512, 512, **symm_config),
            symm.SymmConvSequential(512, 512, pooling=nn.AvgPool2d(2, 2), **symm_config),
            symm.SymmConvSequential(512, 512, **symm_config),
            symm.SymmForward(
                snn=nn.Sequential(snnalgo.Tosnn(nn.BatchNorm2d(512)),
                                  neuron.LIF(tau=tau),
                                  snnalgo.Tosnn(nn.AdaptiveAvgPool2d((7, 7))),),
                ann=nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((7, 7)),),
                symm_training=False, symm_connect=False
            )
        )
        self.classify = symm.SymmForward(
            snn=snnalgo.Tosnn(nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512 * 7 * 7, num_classes))),
            ann=nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512 * 7 * 7, num_classes)),
            symm_training=False, symm_connect=False
        )

    def forward(self, snnx, annx):
        h = self.encod((snnx, annx))
        sh, ah = self.body(h)
        sh = torch.flatten(sh, 2)
        ah = torch.flatten(ah, 1)
        return self.classify((sh, ah))

if __name__ == '__main__':
    import itertools
    import torch
    for symm_training, symm_connect in itertools.product([True, False], [True, False]):
        symm_config = {
            'symm_training': symm_training,
            'symm_connect': symm_connect,
        }
        net = VGG11(symm_config, 3).to(0)
        print(net)
        ax = torch.rand([1, 3, 32, 32]).to(0)
        y = torch.rand(1, 10).to(0)
        sx = snnalgo.temporal_repeat(ax, 2).to(0)
        sout, aout = net(sx, ax)
        loss = torch.nn.functional.mse_loss(aout, y) + torch.nn.functional.mse_loss(sout.mean(dim=1), y)
        loss.backward()
        print()
        print(loss.item())
        print(symm_config)
        print(aout.shape, sout.shape)