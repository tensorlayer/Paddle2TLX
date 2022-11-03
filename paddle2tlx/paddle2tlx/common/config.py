# coding: utf-8
import os
import sys

# sys.path.append(os.path.dirname(__file__))

# path to save converted tlx pretrained model
PRETRAINED_PATH_TLX = "D:/Model/tlx_pretrained_model"
# PRETRAINED_PATH_PD = "C:/Users/Administrator/.cache/paddle/hapi/weights"

# paddle pretrained model download urls
MODEL_PD_URLS = {
    "vgg16": (
        "https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams",
        "89bbffc0f87d260be9b8cdc169c991c4"
    ),
    "vgg19": (
        "https://paddle-hapi.bj.bcebos.com/models/vgg19.pdparams",
        "23b18bb13d8894f60f54e642be79a0dd"
    ),
    "alexnet": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams",
        "7f0f9f737132e02732d75a1459d98a43"
    ),
    'resnet50': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
        'ca6f485ee1ab0492d38f323885b0ad80'
    ),
    'resnet101': (
        'https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
        '02f35f034ca3858e1e54d4036443c92d'
    ),
    "googlenet": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GoogLeNet_pretrained.pdparams",
        "80c06f038e905c53ab32c40eca6e26ae"
    ),
    # mobilenet
    'mobilenetv1_1.0': (
        'https://paddle-hapi.bj.bcebos.com/models/mobilenetv1_1.0.pdparams',
        '3033ab1975b1670bef51545feb65fc45'
    ),
    'mobilenetv2_1.0': (
        'https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0.pdparams',
        '0340af0a901346c8d46f4529882fb63d'
    ),
    "mobilenet_v3_small": (
        "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_small_x1.0.pdparams",
        "34fe0e7c1f8b00b2b056ad6788d0590c"
    ),
    "mobilenet_v3_large": (
        "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_large_x1.0.pdparams",
        "118db5792b4e183b925d8e8e334db3df"
    ),
    # shufflenet
    "shufflenet_v2_x0_25": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_25.pdparams",
        "1e509b4c140eeb096bb16e214796d03b",
    ),
    "shufflenet_v2_x0_33": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_33.pdparams",
        "3d7b3ab0eaa5c0927ff1026d31b729bd",
    ),
    "shufflenet_v2_x0_5": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x0_5.pdparams",
        "5e5cee182a7793c4e4c73949b1a71bd4",
    ),
    "shufflenet_v2_x1_0": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x1_0.pdparams",
        "122d42478b9e81eb49f8a9ede327b1a4",
    ),
    "shufflenet_v2_x1_5": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x1_5.pdparams",
        "faced5827380d73531d0ee027c67826d",
    ),
    "shufflenet_v2_x2_0": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_x2_0.pdparams",
        "cd3dddcd8305e7bcd8ad14d1c69a5784",
    ),
    "shufflenet_v2_swish": (
        "https://paddle-hapi.bj.bcebos.com/models/shufflenet_v2_swish.pdparams",
        "adde0aa3b023e5b0c94a68be1c394b84",
    ),
    #
    'squeezenet1_0': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_0_pretrained.pdparams',
        '30b95af60a2178f03cf9b66cd77e1db1'
    ),
    'squeezenet1_1': (
        'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_1_pretrained.pdparams',
        'a11250d3a1f91d7131fd095ebbf09eee'
    ),
    "inception_v3": (
        "https://paddle-hapi.bj.bcebos.com/models/inception_v3.pdparams",
        "649a4547c3243e8b59c656f41fe330b8"
    ),
    "RegNetX_4GF": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_4GF_pretrained.pdparams",
    ),
    "TNT_small": (
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TNT_small_pretrained.pdparams"
    ),
}
