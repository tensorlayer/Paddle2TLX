import os
os.environ['TL_BACKEND'] = 'paddle'  # don't effect paddle model results before converted
from .conv import tlx_DeformConv2d, tlx_ConvTranspose2d
from .loss import tlx_L1Loss, tlx_BCEWithLogitsLoss, tlx_MSELoss, tlx_cross_entropy
from .norm import tlx_InstanceNorm, tlx_InstanceNorm1d, tlx_InstanceNorm2d, tlx_InstanceNorm3d
from .padding import tlx_Pad2d
from .nn import tlx_Dropout
from .pooling import tlx_MaxPool2d, tlx_AvgPool2d, tlx_MaxUnPool2d
from .sample import tlx_Upsample, tlx_UpsamplingBilinear2d, tlx_Identity#, tlx_PixelShuffle
from .common import tlx_randperm, tlx_get_tensor_shape, tlx_linear, tlx_one_hot, tlx_get_tensor_shape
from .common import tlx_linspace, tlx_nonzero, tlx_instance_norm, tlx_adaptive_avg_pool2d, tlx_adaptive_max_pool2d
from .functional import tlx_hflip, tlx_vflip, tlx_resize, tlx_pad, tlx_crop, tlx_normalize, tlx_to_tensor
from .activation import tlx_GELU
