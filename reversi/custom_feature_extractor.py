
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MyResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        blocks_num:  List[int] = [1],
        in_channels = 4,
        net_arch = [32, 64, 64],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        pool_type=None
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block
        if len(blocks_num) < len(net_arch):
            blocks_num += [1]*(len(net_arch)-len(blocks_num))
        self.blocks_num = blocks_num
        self.inplanes = in_channels
        self.net_arch = net_arch
        self.dilation = 1
        self.groups = groups

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.resnet_block = self._make_resnet_block()

        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        # print(f"dilate:{dilate}, stride:{stride}, block.expansion:{block.expansion}")
        # print(f"inplanes:{inplanes}, planes:{planes}, downsample: {downsample}, ")
        layers = []
        layers.append(
            block(
                inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                groups=self.groups,dilation=previous_dilation, norm_layer=norm_layer
            )
        )
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    groups=self.groups,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return layers

    def _make_resnet_block(self):
        resnet_layers = []
        inplanes = self.inplanes
        for i in range(len(self.net_arch)):
            planes = self.net_arch[i]
            # print(f"inplanes:{inplanes}")
            # print(f"planes:{planes}")
            block_layer = self._make_layer(self.block, inplanes, planes, self.blocks_num[i])
            resnet_layers.extend(block_layer)
            inplanes = planes

        return nn.Sequential(*resnet_layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # print(f"x.shape -1-: {x.shape}")
        x = self.resnet_block(x)
        # print(f"x.shape -2-: {x.shape}")
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(f"x.shape -3-: {x.shape}")
        # x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class MyCnnNet(nn.Module):
    def __init__(
            self,
            in_channels,
            net_arch=[32, 64, 128],
            kernel_size=3,
            stride=1,
            padding=1,
            is_batch_norm=False,
            pool_type=None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.net_arch = net_arch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.is_batch_norm = is_batch_norm
        self.pool_type = pool_type
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size,
                                     stride=stride+1,
                                     padding=padding)
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size,
                                     stride=stride+1,
                                     padding=padding)

        self.conv_block = self._make_layer()

    def _make_layer(self, ) -> nn.Sequential:
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        stride_list = self.stride
        if isinstance(self.stride, int):
            stride_list = [self.stride] * len(self.net_arch)

        kernel_size_list = self.kernel_size
        if isinstance(self.kernel_size, int):
            kernel_size_list = [self.kernel_size] * len(self.net_arch)

        layers = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(len(self.net_arch)):
            out_channels = self.net_arch[i]
            kernel_size = kernel_size_list[i]
            stride = stride_list[i]
            conv_block = [
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=self.padding),
                # nn.BatchNorm2d(num_features=out_channels),
                # nn.ReLU(),
            ]

            if self.is_batch_norm:
                conv_block.append(nn.BatchNorm2d(num_features=out_channels))
            conv_block.append(nn.ReLU())

            # 添加池化层
            if self.pool_type:
                if i == 0 or i == len(self.net_arch)-1:
                    pool = self.max_pool
                    if self.pool_type == 'avg':
                        pool = self.avg_pool
                    conv_block.append(pool)
            in_channels = out_channels
            layers.extend(conv_block)
        layers.extend([nn.Flatten()])

        return nn.Sequential(* layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    # observation_space: spaces.Box
    def __init__(
            self,
            observation_space: spaces.Box,
            features_dim: int = 256,
            net_arch=[32, 64, 128],
            backbone='cnn',
            kernel_size=3,
            stride=1,
            padding=1,
            is_batch_norm=False,
            pool_type=None):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        in_channels = observation_space.shape[0]
        stride_list = stride
        if backbone == 'cnn':
            self.net = self._build_cnn_layers(in_channels, net_arch=net_arch, kernel_size=kernel_size, stride=stride,
                           padding=padding, is_batch_norm=is_batch_norm, pool_type=pool_type)
        else:
            self.net = self._build_resnet_layers(in_channels, net_arch=net_arch, kernel_size=kernel_size, stride=stride,
                           padding=padding, pool_type=pool_type)

        # Compute shape by doing one forward pass
        # 自动推导 n_flatten
        with th.no_grad():
            n_flatten = self.net(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # 手动计算
        # n_flatten = observation_shape[1] * observation_shape[2] * net_arch[-1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def _build_cnn_layers(self, in_channels, net_arch=[32, 64, 128], kernel_size=3,
            stride=1, padding=1, is_batch_norm=False, pool_type=None):
        cnn_net = MyCnnNet(in_channels, net_arch=net_arch, kernel_size=kernel_size, stride=stride,
                           padding=padding, is_batch_norm=is_batch_norm, pool_type=pool_type)
        return cnn_net

    def _build_resnet_layers(self, in_channels, net_arch=[32, 64, 128], kernel_size=3,
            stride=1, padding=1, is_batch_norm=False, pool_type=None):
        res_net = MyResNet(block=BasicBlock, blocks_num=[1,1,1],
            in_channels=in_channels, net_arch=net_arch, num_classes=64, pool_type=pool_type)
        return res_net

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.net(observations))