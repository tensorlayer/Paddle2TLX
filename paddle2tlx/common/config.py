# coding: utf-8
import os


FILE_NOT_CONVERT = [
    "load_model_clas.py", "tlx_extend.py", "tlx_activation.py", "tlx_padding.py",
    "tlx_basic_conv.py", "tlx_basic_norm.py", "tlx_basic_pooling.py", "tlx_basic_sample.py",
    # optimizer: unet, farseg, deeplabv3p
    # "cross_entropy_loss.py", "lovasz_loss.py", "mixed_loss.py",
    # "dice_loss.py", "fccdn_loss.py",
]


FILE_IMPORT_BACKEND = [
    "infer_clas.py", "train_clas.py",
    "infer_det.py", "train_det.py",
    "infer_cdet.py", "train_cdet.py",
    "infer_seg.py", "train_seg.py",
    "infer_gan.py", "train_gan.py",
    "predict_rscd.py", "predict_gan.py", "predict_rsseg.py", "predict_seg.py", "test.py",
]
