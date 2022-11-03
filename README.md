# paddle2tlx

持续更新...

## 功能简介

paddle2tlx 是一款面向 TensorLayerX 的模型转换工具，可以方便的将由 PaddlePaddle 实现的模型迁移到 TensorLayerX 框架上运行。

## 使用描述

### 参数说明

| 参数名称 | 参数描述 |
| -- | -- |
| modle_name | 模型名称 |
| input_dir_pd | paddle源工程目录 |
| output_dst_tlx | tensorlayerx目标工程目录 |
| save_tag | 是否保存转换后的预训练模型 |
| pretrained_model | 转换后预训练模型保存路径 |

### 运行方式

**1. 准备 paddlepaddle 模型工程代码**

将 paddle 模型代码放在自己创建的一个目录下

**1. 修改变量值**

修改 paddle2tlx/common/convert.py 文件 main 方法中的以下几个变量的默认值

- project_src_path：paddle 模型工程代码文件夹
- project_dst_path：转换后 tensorlayerx 模型存放文件夹
- mobile_name：模型名称
- pretrain_model：转换后预训练模型的统一保存路径

**2. 在 pycharm 运行以下命令**

```shell
# 暂不支持
# paddle2tlx --model_name=vgg16 --input_dir_pd=project_src_path --output_dst_tlx=project_dst_path --pretrain_model="D:/Model/tlx_pretrained_model"
python paddle2tlx/convert.py  # pycharm 运行
```

## 更新记录

### 模型适配

| 序号 | 模型 | 类别误差| 前后误差 | 状态 | 方向 | 总表序号 | 完成时间 |
| -- | -- | -- | -- | -- | -- |-- |-- |
| 1 | VGG16(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 31 | 20221028 |
| 2 | AlexNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 1 | 20221101 |
| 3 | ResNet50(pretrained model) | 微小误差 |  |  | PaddleClas | 32 | 20221101 |
| 4 | ResNet101(pretrained model) | 微小误差 |  |  | PaddleClas | 33 | 20221101 |
| 5 | GoogleNet(pretrained model) | 微小误差 |  |  | PaddleClas | 14 | 20221102 |
| 6 | mobilenetv1(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 23 | 20221102 |
| 7 | mobilenetv2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 24 | 20221102 |
| 8 | mobilenetv3(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 25 | 20221103 |
| 9 | shufflenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 39 | 20221103 |
| 10 | squeezenet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 40 | 20221103 |
| 11 | inceptionv3(pretrained model) | 微小误差 |  |  | PaddleClas | 19 | 20221103 |
| 12 | regnet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 30 | 20221103 |


### 功能优化

**20221028**

待补充

## 依赖环境

```shell
python==3.7
paddlepaddle==2.3.0
tensorlayerx==0.5.7  # latest
```
