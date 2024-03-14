# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""

import warnings
import jittor as jt

def grid_sample(input, grid):
    # if _should_use_custom_op():
    # return _GridSample2dForward.apply(input, grid)
    return jt.nn.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)