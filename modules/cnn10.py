import torch.nn as nn
import snetx.snn.algorithm as snnalgo
import snetx.cuend.neuron as neuron
from modules import symm

class CNN10(nn.Module):
    def __init__(self, symm_config, img_channels, h_channels=256, num_classes=10):
        super().__init__()
        self.encod = symm.SymmConvEncoder(
            {'in_channels': img_channels, 'out_channels': h_channels, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            **symm_config
        )
        self.body = nn.Sequential(
            symm.SymmConvSequential(h_channels, h_channels, **symm_config),
            symm.SymmConvSequential(h_channels, h_channels, stride=2, **symm_config),

            symm.SymmConvSequential(h_channels, h_channels, **symm_config),
            symm.SymmConvSequential(h_channels, h_channels, stride=2, **symm_config),

            symm.SymmConvSequential(h_channels, h_channels, **symm_config),
            symm.SymmConvSequential(h_channels, h_channels, stride=2, **symm_config),

            symm.SymmConvSequential(h_channels, h_channels, **symm_config),
            symm.SymmConvSequential(h_channels, h_channels, stride=2, **symm_config),
        )
        classifier = [symm.SymmLinearSequential(h_channels * 4, h_channels, **symm_config),]
        symm_config['symm_training'] = False
        symm_config['symm_connect'] = False
        classifier += [symm.SymmLinearSequential(h_channels, num_classes, **symm_config),]
        self.classifier = nn.Sequential(*classifier)

    def forward(self, snnx, annx):
        snnh, annh = self.encod((snnx, annx))
        sh, ah = self.body((snnh, annh))
        sh = torch.flatten(sh, 2)
        ah = torch.flatten(ah, 1)
        out = self.classifier((sh, ah))
        return out

if __name__ == '__main__':
    import itertools
    import torch
    for symm_training, symm_connect in itertools.product([True, False], [True, False]):
        symm_config = {
            'symm_training': symm_training,
            'symm_connect': symm_connect,
        }
        net = CNN10(symm_config, 3, 256).to(0)
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

    print({}.get(1, 2))

