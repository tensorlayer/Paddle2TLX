# paddle2tlx

持续更新...

## 功能简介

paddle2tlx 是一款面向 TensorLayerX 的模型转换工具，可以方便的将由 PaddlePaddle 实现的模型迁移到 TensorLayerX 框架上运行。

## 使用描述

### 参数说明

| 参数名称 | 参数描述 |  备注 |
| -- | -- |  -- |
| modle_name | 模型名称 | 可选(便于转换后验证模型) |
| model_type | 模型类别 | 可选(便于转换后验证模型) |
| input_dir_pd | paddle源工程目录 | 必选 |
| output_dir_tlx | tensorlayerx目标工程目录 | 必选 |
| save_tag | 是否保存转换后的预训练模型 | 模型脚本中设置 |
| pretrained_model | 转换后预训练模型保存路径 | 模型脚本中设置 |

### 使用方式

#### 准备转前 paddle 模型工程代码

将 paddle 模型代码放在自己创建的一个文件夹下，转前和转后代码按模型任务类别进行划分，文件夹包含当前模型源码文件和依赖代码。预训练模型权重存放路径采用的是脚本指定的外部文件夹。以图像分类模型为例，转前模型文件夹结构如下，其中包含60个分类任务的模型定义脚本：

```shell
pd_models/  # 转前模型根目录
└── paddleclas  # 图像分类模型
    ├── alexnet.py
    ├── convnext.py
    ├── cspdarknet.py
    ├── cswin_transformer.py
    ├── darknet53.py
    ├── deit.py
    ├── __init__.py
    ├── ops # 依赖算子
    │   ├── __init__.py
    │   ├── ops_fusion.py
    │   └── theseus_layer.py
    └── utils  # 通用方法
        ├── common_func.py
        └── __init__.py
    ...
```

#### 执行转换

**Pycharm中执行转换**

改变 paddle2tlx/convert.py 文件以下几个变量的默认值，然后运行该文件执行转换。

- input_dir_pd：paddle 模型工程代码文件夹
- output_dir_tlx：转换后 tensorlayerx 模型存放文件夹
- model_name：模型名称(可选)
- model_type：模型类别(可选)

**命令行方式转换**

```shell
# 1. 首先, 将 paddle2tlx 工具包装到自己创建的 Python 环境中
pip install -e .

# 2. 然后, 执行代码转换
# 方式1 - 推荐
# 先执行代码转换
paddle2tlx --input_dir_pd pd_models/paddleclas --output_dir_tlx tlx_models/paddleclas
# 转换后单独验证模型
cd examples
python validation.py --input_dir_pd ../pd_models/paddleclas --output_dir_tlx ../tlx_models/paddleclas --model_name vgg16 --model_type clas

# 方式2
# 转换+验证某个模型
paddle2tlx --input_dir_pd pd_models/paddleclas --output_dir_tlx tlx_models/paddleclas --model_name vgg16 --model_type clas
```

转换后模型的目录结构和转前目录结构保持一致。不同任务类别模型的训练脚本和测试脚本存放在 examples 目录下，可留作单独测试用。不同设备间迁移测试模型时，可以保留 pd_models 和 tlx_models 下的模型工程代码和 examples 目录。

对于不支持的 API 算子，还需对转换工具做适配优化。

#### 预训练模型存放路径

如果预训练模型提供了下载链接，会将预训练模型自动下载到 `~/.cache/paddle/hapi/weights` 目录，不同任务存放在各自的子文件夹下，如分类模型会自动下载到 `~/.cache/paddle/hapi/weights/paddleclas` 目录。

如果预训练模型没有提供下载链接或是自己训练，会将预训练模型统一下载到 `pretrain` 目录下，同样按任务区分。


## 更新记录

### 模型适配

#### 分类模型

| 序号 | 模型 | 类别误差 | 前后误差 | 状态 | 方向 | 总表序号 |
| -- | -- | -- | -- | -- | -- |-- |
| 1 | vgg16(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 31 |
| 2 | alexnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 1 |
| 3 | resnet50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 32 |
| 4 | resnet101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 33 |
| 5 | googlenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 14 |
| 6 | mobilenetv1(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 23 |
| 7 | mobilenetv2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 24 |
| 8 | mobilenetv3(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 25 |
| 9 | shufflenetv2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 39 |
| 10 | squeezenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 40 |
| 11 | inceptionv3(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 19 |
| 12 | regnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 30 |
| 13 | tnt(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 41 |
| 14 | darknet53(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 46 |
| 15 | densenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 5 |
| 16 | rednet50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 28 |
| 17 | rednet101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 29 |
| 18 | cspdarknet53(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 2 |
| 19 | efficientnet_b1(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 11 |
| 20 | efficientnet_b7(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 12 |
| 21 | dla34(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 7 |
| 22 | dla102(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 8 |
| 23 | dpn68(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 9 |
| 24 | dpn107(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 10 |
| 25 | ghostnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 13 |
| 26 | hardnet39(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 17 |
| 27 | hardnet85(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 18 |
| 28 | resnest50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 34 |
| 29 | resnext50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 35 |
| 30 | resnext101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 36 |
| 31 | rexnet(pretrained model) | 微小误差 | 0.00061244145 | 完成 | PaddleClas | 37 |
| 32 | se_resnext(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 38 |
| 33 | esnet_x0_5(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 47 |
| 34 | esnet_x1_0(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 48 |
| 35 | vit(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas |  |
| 36 | alt_gvt_small(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 15 |
| 37 | alt_gvt_base(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 16 |
| 38 | swin_transformer_base(pretrained model) | 0.0 |  |  | PaddleClas | 3 |
| 39 | swin_transformer_small(pretrained model) | 0.0 |  |  | PaddleClas | 4 |
| 40 | pcpvt_base(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 26 |
| 41 | pcpvt_large(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 27 |
| 42 | xception41(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 42 |
| 43 | xception65(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 43 |
| 44 | xception41_deeplab(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 44 |
| 45 | xception65_deeplab(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 45 |
| 46 | levit(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 21 |
| 47 | mixnet(pretrained model) | 微小误差 | 0.00048300158 | 完成 | PaddleClas | 22 |
| 48 | convnext(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 49 | cswin(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 50 | deittiny(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 51 | deitsmall(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 52 | deitbase(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 53 | dvt(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 54 | peleenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 55 | pp_hgnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 56 | pp_lcnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 57 | pp_lcnet_v2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 58 | pvt_v2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 59 | res2net(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |
| 60 | van(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 新增 |


#### 分割模型

| 序号 | 模型 | 前后误差 | 状态 | 方向 | 总表序号 |
| -- | -- | -- | -- | -- |-- |
| 1 | fast_scnn | 0.0 | 完成 | PaddleSeg | 76 |
| 2 | hrnet | 0.0 | 完成 | PaddleSeg | 75 |
| 3 | encnet | 0.0 | 完成 | PaddleSeg | 77 |
| 4 | bisenet | 0.0 | 完成 | PaddleSeg | 83 |
| 5 | fastfcn | 0.0 | 完成 | PaddleSeg | 73 |
| 6 | enet | 0.0 | 完成 | PaddleSeg | 74 |


#### 检测模型

| 序号 | 模型 | 前后误差 | 状态 | 方向 | 总表序号 |
| -- | -- | -- | -- | -- |-- |
| 1 | yolov3 | 0.0 | 完成 | PaddleDec | 54 |
| 2 | ssd | 0.0 | 完成 | PaddleDec | 60 |
| 3 | yolox | 0.0 | 完成 | PaddleDec | 57 |
| 4 | picodet_lcnet | 0.0 | 完成 | PaddleDec | 67 |
| 5 | fcos_r50 | 0.0 | 完成 | PaddleDec | 68 |
| 6 | fcos_dcn | 0.0 | 完成 | PaddleDec | 69 |
| 7 | RetinaNet | 0.0 | 完成 | PaddleDec | 64 |
| 8 | Mask_RCNN | 0.0 | 完成 | PaddleDec | 63 |
| 9 | Faster_RCNN | 0.0 | 完成 | PaddleDec | 62 |
| 10 | CascadeRCNN | 0.0 | 完成 | PaddleDec | 70 |
| 11 | SOLOv2 | 0.0 | 完成 | PaddleDec | 72 |
| 12 | GFL | 0.0 | 完成 | PaddleDec | 新增 |
| 13 | TOOD | 0.0 | 完成 | PaddleDec | 新增 |
| 14 | CenterNet | 0.0 | 完成 | PaddleDec | 新增 |
| 15 | TTFNet | 0.0 | 完成 | PaddleDec | 新增 |


#### 遥感模型

| 序号 | 模型 | 前后误差 | 状态 | 方向 | 总表序号 |
| -- | -- | -- | -- | -- |-- |
| 1 | bit | 0.0 | 完成 | PaddleRS | 98 |
| 2 | cdnet | 0.0 | 完成 | PaddleRS | 87 |
| 3 | stanet | 0.0 | 完成 | PaddleRS | 88 |
| 4 | fcef | 0.0 | 完成 | PaddleRS | 89 |
| 5 | fccdn | 0.0 | 完成 | PaddleRS | 91 |
| 6 | dsamnet | 0.0 | 完成 | PaddleRS | 97 |
| 7 | snunet | 0.0 | 完成 | PaddleRS | 90 |
| 8 | dsifn | 0.0 | 完成 | PaddleRS | 95 |
| 9 | unet | 0.0 | 完成 | PaddleRS | 84 |
| 10 | farseg | 0.0 | 完成 | PaddleRS | 85 |
| 11 | deeplab | 0.0 | 完成 | PaddleRS | 86 |


#### 生成模型

| 序号 | 模型 | 前后误差 | 状态 | 方向 | 总表序号 |
| -- | -- | -- | -- | -- |-- |
| 1 | cyclegan | 0.0 | 完成 | PaddleGAN | 78 |
| 2 | starganv2 | 0.0 | 完成 | PaddleGAN | 80 |
| 3 | prenet | 0.0 | 完成 | PaddleGAN | 81 |
| 4 | u-gat-it | 0.0 | 完成 | PaddleGAN | 82 |
| 5 | styleganv2 | 0.0 | 完成 | PaddleGAN | 79 |


### 自然语言模型

| 序号 | 模型 | 前后误差 | 状态 | 方向 | 总表序号 |
| -- | -- | -- | -- | -- |-- |
| 1 | TextCNN | 0.0 | 完成 | PaddleNLP | 100 |
| 2 | LSTM | 0.0 | 完成 | PaddleNLP | 93 |
| 3 | RNN | 0.0 | 完成 | PaddleNLP | 94 |


### 功能优化

**20221028**

待补充


## 依赖环境

```shell
python=3.7
paddlepaddle==2.3.0
tensorlayerx==0.5.7  # latest
```

详见 requirements.txt 文件


## 参考

- [TensorLayerX](https://github.com/tensorlayer/TensorLayerX)
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
