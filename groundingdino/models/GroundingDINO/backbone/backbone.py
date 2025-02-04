# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

from typing import List

from torch import nn

from groundingdino.util.misc import NestedTensor

from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=FrozenBatchNorm2d,
        )
        bb_num_channels = backbone.num_channels
    elif args.backbone in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )

        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, List
    ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    # import ipdb; ipdb.set_trace()
    return model
