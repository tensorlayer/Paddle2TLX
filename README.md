## paddle2tlx-vision

持续更新...

### 功能描述

利用 TensorLayerX 框架复现 PaddlePaddle 内置图像分类模型，对转换后模型做训练和预测验证

### 实现方式

手动替换算子

### 误差计算公式

L1 距离取绝对值


**推理预测误差**

| 序号 | 模型 | 类别误差| 前后误差 |状态|方向|总表序号|完成时间|
| -- | -- | -- | -- | -- | -- |-- |-- |
| 1 | AlexNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 1 | 20221024 |
| 2 | CSPDarknet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 2 | 20221021 
| 3 | DenseNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 5 | 20221016 |
| 4 | DLA34(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 7 | 20221021 |
| 5 | DLA102(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 8 | 20221021 |
| 6 | DNP68(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 9 | 20221021 |
| 7 | DNP107(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 10 | 20221021 |
| 8 | EfficentNetB1(pretrained model) |一致 | 0.0 | 完成 | PaddleClas | 11 | 20221021 |
| 9 | EfficentNetB7(pretrained model) |一致 | 0.0 | 完成 | PaddleClas | 12 | 20221021 |
| 10 | GhostNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 13 | 20221021 |
| 11 | GoogleNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 14 | 20221021 |
| 12 | HardNet39(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 17 | 20221021 |
| 13 | HardNet85(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 18 |  20221021 |
| 14 | Inceptionv3(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 19 | 20221024 |
| 15 | MobileNetV1(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 23 | 20221016 |
| 16 | MobileNetV2(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 24 | 20221016 |
| 17 | MobileNetV3(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 25 | 20221022 |
| 18 | RedNet50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 28 | 20221016 |
| 19 | RedNet101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 29 | 20221016 |
| 20 | VGG16(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 31 | 20221016 |
| 21 | ResNet50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 32 | 20221016 |
| 22 | ResNet101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 33 | 20221016 |
| 23 | ResNest(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 34 | 20221021 | 
| 24 | ResNext50(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 35 | 20221021 |
| 25 | ResNext101(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 36 | 20221021 |
| 26 | RexNet(pretrained model) | 一致 | sum value:0.1267,max value:0.0468 | 完成 | PaddleClas | 37 | 20221021 | 
| 27 | SE-ResNeXt(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 38 | 20221021 |
| 28 | ShuffleNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 39 | 20221016 | 
| 29 | SqueezeNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 40 | 20221024 | 
| 30 | DarkNet(pretrained model) | 一致 | 0.0 | 完成 | PaddleClas | 46 | 20221016 |

**网络模型**
paddle网络模型在目录models_pd， tensorlayerx网络模型在目录models_tlx

**训练模型**

paddle参见文件 1-train_class_pd.py，可加载预训练模型做参数微调，也可从头开始训练
tensorlayerx参见文件 1-train_class_tlx.py，可加载预训练模型做参数微调，也可从头开始训练

**推理误差测试**

参见文件 3-tests_inference_diff.py


### 参考

- [TensorLayerX](https://github.com/tensorlayer/TensorLayerX)
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
- [TensorLayerX-api](https://tensorlayerx.readthedocs.io/en/latest/)
- [PaddlePaddle-api](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
