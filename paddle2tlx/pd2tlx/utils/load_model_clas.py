# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import warnings
import six
import wget
import pickle
import paddle
import numpy as np
from tensorlayerx.files import assign_weights
from .load_model import get_path_from_url, tlx_load, get_new_key
warnings.filterwarnings("ignore")


# mode_urls = {
#     "vgg16": "https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams",
#     "alexnet": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams",
#     'resnet50': 'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
#     'resnet101': 'https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
#     "googlenet": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GoogLeNet_pretrained.pdparams",
#     # mobilenet
#     'mobilenetv1_1.0': 'https://paddle-hapi.bj.bcebos.com/models/mobilenetv1_1.0.pdparams',
#     'mobilenetv2_1.0': 'https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0.pdparams',
#     "mobilenet_v3_small_x1.0": "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_small_x1.0.pdparams",
#     "mobilenet_v3_large_x1.0": "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_large_x1.0.pdparams",
#     # shufflenet
#     "shufflenet_v2_x0_25": "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_25.pdparams",
#     "shufflenet_v2_x0_33": "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_33.pdparams",
#     "shufflenet_v2_x0_5": "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_5.pdparams",
#     "shufflenet_v2_x1_0": "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x1_0.pdparams",
#     "shufflenet_v2_x1_5": "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x1_5.pdparams",
#     "shufflenet_v2_x2_0": "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x2_0.pdparams",
#     "shufflenet_v2_swish": "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_swish.pdparams",
#     # squeezenet
#     'squeezenet1_0': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_0_pretrained.pdparams',
#     'squeezenet1_1': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_1_pretrained.pdparams',
#     "inception_v3": "https://paddle-hapi.bj.bcebos.com/models/inception_v3.pdparams",
#     "regnetx_4gf": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_4GF_pretrained.pdparams",
#     "tnt_small": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TNT_small_pretrained.pdparams",
#     "darknet53": 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams',
#     # densenet
#     'densenet121': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams',
#     'densenet161': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams',
#     'densenet169': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams',
#     'densenet201': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams',
#     'densenet264': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams',
#     # rednet
#     "rednet26": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet26_pretrained.pdparams",
#     "rednet38": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet38_pretrained.pdparams",
#     "rednet50": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet50_pretrained.pdparams",
#     "rednet101": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet101_pretrained.pdparams",
#     "rednet152": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet152_pretrained.pdparams",
#     "cspdarknet53": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSPDarkNet53_pretrained.pdparams",
#     # dla
#     "dla34": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA34_pretrained.pdparams",
#     "dla102": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102_pretrained.pdparams",
#     # dpn
#     "dpn68": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN68_pretrained.pdparams",
#     "dpn107": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN107_pretrained.pdparams",
#     # efficientnet
#     "efficientnet_b1": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB1_pretrained.pdparams",
#     "efficientnet_b7": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB7_pretrained.pdparams",
#     # ghostnet
#     "ghostnet_x0_5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x0_5_pretrained.pdparams",
#     "ghostnet_x1_0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams",
#     "ghostnet_x1_3": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_pretrained.pdparams",
#     # hardnet
#     'hardnet39_ds': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet39_ds_pretrained.pdparams',
#     'hardnet85': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet85_pretrained.pdparams',
#     # resnest
#     "resnest50_fast_1s1x64d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_fast_1s1x64d_pretrained.pdparams",
#     "resnest50": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_pretrained.pdparams",
#     "resnest101": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt101_pretrained.pdparams",
#     # resnext
#     "resnext50_32x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_32x4d_pretrained.pdparams",
#     "resnext50_64x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_64x4d_pretrained.pdparams",
#     "resnext101_32x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x4d_pretrained.pdparams",
#     "resnext101_64x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_64x4d_pretrained.pdparams",
#     "resnext152_32x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_32x4d_pretrained.pdparams",
#     "resnext152_64x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_64x4d_pretrained.pdparams",
#     # rexnet
#     "rexnet_1_0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_0_pretrained.pdparams",
#     "rexnet_1_3": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_3_pretrained.pdparams",
#     "rexnet_1_5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_5_pretrained.pdparams",
#     "rexnet_2_0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_2_0_pretrained.pdparams",
#     "rexnet_3_0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_3_0_pretrained.pdparams",
#     # se_resnext
#     "se_resnext50_32x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_32x4d_pretrained.pdparams",
#     "se_resnext101_32x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt101_32x4d_pretrained.pdparams",
#     "se_resnext152_64x4d": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt152_64x4d_pretrained.pdparams",
#     # esnet
#     "esnet_x0_25": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_25_pretrained.pdparams",
#     "esnet_x0_5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_5_pretrained.pdparams",
#     "esnet_x0_75": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_75_pretrained.pdparams",
#     "esnet_x1_0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x1_0_pretrained.pdparams",
#     # vit
#     "vit_small_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams",
#     "vit_base_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams",
#     "vit_base_patch16_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams",
#     "vit_base_patch32_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams",
#     "vit_large_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams",
#     "vit_large_patch16_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_384_pretrained.pdparams",
#     "vit_large_patch32_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch32_384_pretrained.pdparams",
#     # gvt
#     "pcpvt_small": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_small_pretrained.pdparams",
#     "pcpvt_base": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_base_pretrained.pdparams",
#     "pcpvt_large": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_large_pretrained.pdparams",
#     "alt_gvt_small": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_small_pretrained.pdparams",
#     "alt_gvt_base": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_base_pretrained.pdparams",
#     "alt_gvt_large": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_large_pretrained.pdparams",
#     # swintransformer
#     "swintransformer_tiny_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams",
#     "swintransformer_small_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_small_patch4_window7_224_pretrained.pdparams",
#     "swintransformer_base_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window7_224_pretrained.pdparams",
#     "swintransformer_base_patch4_window12_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window12_384_pretrained.pdparams",
#     "swintransformer_large_patch4_window7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window7_224_22kto1k_pretrained.pdparams",
#     "swintransformer_large_patch4_window12_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window12_384_22kto1k_pretrained.pdparams",
#     # xception
#     "xception41": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_pretrained.pdparams",
#     "xception65": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_pretrained.pdparams",
#     "xception71": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception71_pretrained.pdparams",
#     # xception_deeplab
#     "xception41_deeplab": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_deeplab_pretrained.pdparams",
#     "xception65_deeplab": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_deeplab_pretrained.pdparams",
#     # levit
#     "levit_128s": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128S_pretrained.pdparams",
#     "levit_128": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128_pretrained.pdparams",
#     "levit_192": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_192_pretrained.pdparams",
#     "levit_256": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_256_pretrained.pdparams",
#     "levit_384": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_384_pretrained.pdparams",
#     # mixnet
#     "mixnet_s": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_S_pretrained.pdparams",
#     "mixnet_m": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_M_pretrained.pdparams",
#     "mixnet_l": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_L_pretrained.pdparams",
#     # new added model
#     "ConvNeXt_tiny": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_tiny_pretrained.pdparams",
#     "CSWinTransformer_tiny_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_tiny_224_pretrained.pdparams",
#     "DeiT_tiny_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_patch16_224_pretrained.pdparams",
#     "DeiT_small_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_patch16_224_pretrained.pdparams",
#     "DeiT_base_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams",
#     "DeiT_base_distilled_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_224_pretrained.pdparams",
#     # peleenet
#     "PeleeNet": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PeleeNet_pretrained.pdparams",
#     # pp_hgnet
#     "PPHGNet_tiny": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_tiny_pretrained.pdparams",
#     # pp_lcnet
#     "PPLCNet_x0_25": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
#     # pp_lcnetv2
#     "PPLCNetV2_base": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNetV2_base_pretrained.pdparams",
#     # pvtv2
#     "PVT_V2_B0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B0_pretrained.pdparams",
#     # res2net
#     "Res2Net50_26w_4s": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_26w_4s_pretrained.pdparams",
#     # van
#     "VAN_B0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VAN_B0_pretrained.pdparams",
#     # TODO
# }


def get_param_info(model):
    total_params = 0
    trainable_params = 0
    nontrainable_params = 0
    i = 0
    for param_name, layer_p in model.named_parameters():
        print(f"{i+1}\t\t{param_name}\t\t{layer_p.name}\t\t{layer_p.shape}")  # print model layer name
        volume = np.prod(layer_p.shape)
        total_params += volume
        if layer_p.stop_gradient:
            nontrainable_params += volume
        else:
            trainable_params += volume
        i += 1
    print(f'Total params: {total_params}')
    print(f'Trainable params: {trainable_params}')
    print(f'Non-trainable params: {nontrainable_params}')
    return total_params, trainable_params, nontrainable_params


def load_model_clas(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddleclas")
    param = paddle.load(weights_path)
    param = get_new_key(param)  # todo
    model.load_dict(param)
    return model


def restore_model_clas(model, model_name, urls):
    _, weights_path = get_path_from_url(urls, model_name, "paddleclas")
    param = tlx_load(weights_path)
    restore_model(param, model)
    return model


"""
classification model
"""
# def restore_model(param, model):  # convnext
#     tlx2pd_namelast = {'filters': 'weight',        # conv2d
#                        'biases': 'bias',           # linear
#                        'weights': 'weight',        # linear
#                        'gamma': 'weight',          # bn, ln
#                        'beta': 'bias',             # bn, ln
#                        'moving_mean': '_mean',     # bn
#                        'moving_var': '_variance',  # bn
#                        }
#     model_state = [i for i, k in model.named_parameters()]
#     weights = []
#     # get_param_tlx(model)
#     # [print(i, k.shape) for i, k in model.named_parameters()]
#     for i in range(len(model_state)):
#         model_key = model_state[i]
#         model_keys = model_key.rsplit('.', 1)
#         if len(model_keys) == 2 and model_key not in param:
#             if model_keys[1] in tlx2pd_namelast:
#                 model_key = model_keys[0] + '.' + tlx2pd_namelast[model_keys[1]]
#             else:
#                 model_key = model_key
#         weights.append(param[model_key])
#     assign_weights(weights, model)
#     # save_file = os.path.join(PRETRAINED_PATH_TLX, arch + ".npz")
#     # tlx.files.save_npz(model.all_weights, name=save_file)  # save model
#     del weights
#     return model


# def restore_model(params, model):
#     tlx2pd_namelast = {'filters': 'weight',        # conv2d
#                        'biases': 'bias',           # linear
#                        'weights': 'weight',        # linear
#                        'gamma': 'weight',          # bn, ln
#                        'beta': 'bias',             # bn, ln
#                        'moving_mean': '_mean',     # bn
#                        'moving_var': '_variance',  # bn
#                        }
#     model_states = [i for i, k in model.named_parameters()]
#     # [print(k, v.shape) for k, v in model.named_parameters()]
#     # [print(k, v.shape) for k, v in params.items()]
#
#     weights = []
#     for model_k, model_v in model.named_parameters():
#         model_key_split = model_k.rsplit('.', 1)
#         if len(model_key_split) == 2 and model_key_split[1] in tlx2pd_namelast:
#             param_k = model_key_split[0] + '.' + tlx2pd_namelast[model_key_split[1]]
#             # assert model_v.shape == params[param_k].shape
#             # weights.append(params[param_k])
#             # print('----------------------')
#             # print(model_k, model_v.shape)
#             # print(param_k, params[param_k].shape)
#             try:
#                 weights.append(params[param_k])
#                 print('----------------------')
#                 print(model_k, model_v.shape)
#                 print(param_k, params[param_k].shape)
#             except KeyError as err:
#                 warnings.warn(("Skip loading for {}. ".format(param_k) + str(err)))
#         elif model_k in params:
#             param_k = model_k
#             # assert model_v.shape == params[param_k].shape
#             weights.append(params[param_k])
#             print('model_key == param key, key is: ', param_k)
#             print(model_k, model_v.shape)
#             print(param_k, params[param_k].shape)
#         else:
#             print('unmatched model key: ', model_k)
#
#     print('len model states:', len(model_states), ' len params:', len(params), 'len matched weights:', len(weights))
#     assign_weights(weights, model)
#     del weights
#     return model


def restore_model(param, model):
    tlx2pd_namelast = {'filters': 'weight',        # conv2d
                       'biases': 'bias',           # linear
                       'weights': 'weight',        # linear
                       'gamma': 'weight',          # bn, ln
                       'beta': 'bias',             # bn, ln
                       'moving_mean': '_mean',     # bn
                       'moving_var': '_variance',  # bn
                       # custom op name
                       'batch_norm.gamma': '_batch_norm.weight',
                       'batch_norm.beta': '_batch_norm.bias',
                       'batch_norm.moving_mean': '_batch_norm._mean',
                       'batch_norm.moving_var': '_batch_norm._variance',
                       }
    model_state = [i for i, k in model.named_parameters()]
    weights = []
    # get_param_info(model)

    for i in range(len(model_state)):
        model_key = model_state[i]
        model_keys = model_key.rsplit('.', 1)
        indices = [i for i, c in enumerate(model_key) if c == '.']
        model_key_l, model_key_r = '', ''
        if len(indices) >= 2:
            model_key_l = model_key[:indices[-2]]
            model_key_r = model_key[indices[-2]+1:]
        if len(model_keys) == 2 and model_key not in param:  # if len(model_keys) == 2:
            if model_keys[1] in tlx2pd_namelast and model_key_r not in tlx2pd_namelast:
                model_key = model_keys[0] + '.' + tlx2pd_namelast[model_keys[1]]
            elif model_keys[1] in tlx2pd_namelast and model_key_r in tlx2pd_namelast:
                model_key = model_key_l + '.' + tlx2pd_namelast[model_key_r]
            else:
                model_key = model_key
        weights.append(param[model_key])
    assign_weights(weights, model)
    # save_file = os.path.join(PRETRAINED_PATH_TLX, arch + ".npz")
    # tlx.files.save_npz(model.all_weights, name=save_file)  # save model
    del weights
