# Based on https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py #

from typing import List, Optional, Dict
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

import models.mobilenetv3 as mobilenetv3
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation.fcn import FCNHead


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [4, 8, 12]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x,
                             size=size,
                             mode="bilinear",
                             align_corners=False)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 atrous_rates: List[int],
                 out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels,
                      out_channels,
                      1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    __constants__ = ["aux_classifier"]

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None,
        grid_mode: bool = False,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.grid_mode = grid_mode

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)
        if not self.grid_mode:
            x = F.interpolate(x,
                              size=input_shape,
                              mode="bilinear",
                              align_corners=False)

        if self.aux_classifier is not None:
            result = OrderedDict()
            result["out"] = x

            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x,
                              size=input_shape,
                              mode="bilinear",
                              align_corners=False)

            result["aux"] = x
        else:
            result = x

        return result


def deeplabv3_mobilenet_v3(
    small: bool = True,
    num_classes: int = 22,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = False,
    reduced_tail: bool = True,
    grid_mode: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """

    backbone = mobilenetv3.mobilenet_v3(small=small,
                                        pretrained=pretrained_backbone,
                                        dilated=True,
                                        reduced_tail=reduced_tail)

    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [
        i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)
    ] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux_loss:
        return_layers[str(aux_pos)] = "aux"
    backbone = create_feature_extractor(backbone, return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux_loss else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier, grid_mode)
