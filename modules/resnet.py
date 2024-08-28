from functools import partial
from typing import Any, Callable, List, Optional, Dict, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from snetx.cuend import neuron
from snetx.snn import algorithm as snnalgo
import symm

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
            self,
            symm_config,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride
        self.symm1 = symm.SymmConvSequential(inplanes, planes, stride, **symm_config)
        self.symm2 = symm.SymmConvSequential(planes, planes, **symm_config)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(identity)
        h = self.symm1(x)
        h = self.symm2(h)
        return [h[0] + identity[0], h[1] + identity[1]]


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            symm_config,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        tau = symm_config.get('tau', 2.0)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride
        self.symm1 = symm.SymmForward(
            snn=nn.Sequential(snnalgo.Tosnn(nn.BatchNorm2d(inplanes)), neuron.LIF(tau=tau),
                              snnalgo.Tosnn(nn.Conv2d(inplanes, width, 1, 1))),
            ann=nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(), nn.Conv2d(inplanes, width, 1, 1,)),
            **symm_config
        )
        self.symm2 = symm.SymmForward(
            snn=nn.Sequential(
                snnalgo.Tosnn(nn.BatchNorm2d(width)),
                neuron.LIF(tau=tau),
                snnalgo.Tosnn(nn.Conv2d(width, width, 3, stride, groups=groups, dilation=dilation))),
            ann=nn.Sequential(
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.Conv2d(width, width, 3, stride, groups=groups, dilation=dilation),),
            **symm_config
        )
        self.symm3 = symm.SymmForward(
            snn=nn.Sequential(
                snnalgo.Tosnn(nn.BatchNorm2d(width)),
                neuron.LIF(tau=tau),
                snnalgo.Tosnn(nn.Conv2d(width, planes * self.expansion, 1, 1,))),
            ann=nn.Sequential(
                nn.BatchNorm2d(width),
                nn.ReLU(), 
                nn.Conv2d(width, planes * self.expansion, 1, 1,)),
            **symm_config
        )

    def forward(self, x):
        identity = x

        if self.downsample:
            identity = self.downsample(identity)
        
        h = self.symm1(x)
        h = self.symm2(h)
        h = self.symm3(h)
        return [h[0] + identity[0], h[1] + identity[1]]


class ResNet(nn.Module):
    def __init__(
            self,
            symm_config,
            block,
            layers: List[int],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        if num_classes == 1000:
            conv_config = {
                'in_channels': 3, 'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3}
            self.feature = symm.SymmConvEncoder(conv_config, **symm_config)
        else:
            conv_config = {
                'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            self.feature = symm.SymmConvEncoder(conv_config, **symm_config)

        self.layer1 = self._make_layer(symm_config, block, 64, layers[0])
        self.layer2 = self._make_layer(symm_config, block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(symm_config, block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(symm_config, block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        tau = symm_config.get('tau', 2.0)
        self.pooling = symm.SymmForward(
            snn=nn.Sequential(snnalgo.Tosnn(nn.BatchNorm2d(512 * block.expansion)),
                              neuron.LIF(tau=tau), snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1)))),
            ann=nn.Sequential(nn.BatchNorm2d(512 * block.expansion),
                              nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))),
            symm_training=False, symm_connect=False
        )
        self.classify = symm.SymmForward(
            snn=snnalgo.Tosnn(nn.Linear(512 * block.expansion, num_classes)),
            ann=nn.Linear(512 * block.expansion, num_classes),
            symm_training=False, symm_connect=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            symm_config,
            block,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        tau = symm_config.get('tau', 2.0)
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = symm.SymmForward(
                snn=nn.Sequential(
                    neuron.LIF(tau=tau),
                    snnalgo.Tosnn(nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                        nn.BatchNorm2d(planes * block.expansion),))),
                ann=nn.Sequential(
                    nn.ReLU(),
                    nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                        nn.BatchNorm2d(planes * block.expansion))),
                **symm_config
            )

        layers = []
        layers.append(
            block(
                symm_config, self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                previous_dilation
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    symm_config,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.feature(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        s, a = self.pooling(x)
        s = torch.flatten(s, 2)
        a = torch.flatten(a, 1)
        x = self.classify((s, a))

        return x

    def forward(self, snnx, annx):
        return self._forward_impl((snnx, annx))


def _resnet(
        symm_config,
        block,
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    model = ResNet(symm_config, block, layers, **kwargs)

    return model


V = TypeVar('V')


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def resnet18(symm_config, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """

    return _resnet(symm_config, BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(symm_config, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    return _resnet(symm_config, BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(symm_config, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """

    return _resnet(symm_config, Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(symm_config, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """
    return _resnet(symm_config, Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(symm_config, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """

    return _resnet(symm_config, Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(symm_config, **kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(symm_config, Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(symm_config, **kwargs: Any) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(symm_config, Bottleneck, [3, 4, 23, 3], **kwargs)


def resnext101_64x4d(symm_config, **kwargs: Any) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(symm_config, Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(symm_config, **kwargs: Any) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(symm_config, Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(symm_config, **kwargs: Any) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(symm_config, Bottleneck, [3, 4, 23, 3], **kwargs)

if __name__ == '__main__':
    import itertools
    import torch
    for symm_training, symm_connect in itertools.product([True, False], [True, False]):
        symm_config = {
            'symm_training': symm_training,
            'symm_connect': symm_connect,
        }
        net = resnet18(symm_config, num_classes=10).to(0)
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