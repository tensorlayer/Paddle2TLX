# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import paddle
import tensorlayerx as tlx
import numpy as np
from utils.options import parse_args
from models_tlx import tlx_alexnet
from models_tlx import tlx_cspdarknet
from models_tlx import tlx_densenet
from models_tlx import tlx_dla
from models_tlx import tlx_dpn
from models_tlx import tlx_efficientnet
from models_tlx import tlx_ghostnet
from models_tlx import tlx_googlenet
from models_tlx import tlx_hardnet
from models_tlx import tlx_inceptionv3
from models_tlx import tlx_mobilenetv1, tlx_mobilenetv2, tlx_mobilenetv3
from models_tlx import tlx_rednet
from models_tlx import tlx_vgg
from models_tlx import tlx_resnet
from models_tlx import tlx_resnest
from models_tlx import tlx_resnext
from models_tlx import tlx_resnext
from models_tlx import tlx_rexnet
from models_tlx import tlx_se_resnext
from models_tlx import tlx_shufflenetv2
from models_tlx import tlx_squeezenet
from models_tlx import tlx_darknet53

# EPOCH_NUM = 10
EPOCH_NUM = 4
BATCH_SIZE = 8  # 64
BATCH_NUM = 4  # 100
IMAGE_SHAPE = [3, 224, 224]
CLASS_NUM = 1000


class RandomDataset(tlx.dataflow.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random(size=IMAGE_SHAPE).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class CatDogDataset(tlx.dataflow.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, item):
        pass

    def __len__(self):
        return self.num_samples


class ModelTrainTLX(object):
    def __init__(self, model):
        self.model = model

    def train(self, model_name="VGG16"):
        import paddle.nn as nn
        import paddle.optimizer as opt
        loss_fn = nn.CrossEntropyLoss()
        adam = opt.Adam(learning_rate=0.001, parameters=model.parameters())

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=0)  # 2
        print(f"{args.model_name} begin...")
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader()):
                out = model(image)
                if isinstance(out,list):
                    out = out[0]
                # print(f"type(out)={type(out)}")
                loss = loss_fn(paddle.to_tensor(out), label)
                loss.backward()
                adam.step()
                adam.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(epoch_id, batch_id, np.mean(loss.numpy())))
                # for key in self.model.state_dict().keys():
                #     print(key, self.model.state_dict()[key].shape)
                exit()
        print(f"{args.model_name} end...")

import  models_tlx
if __name__ == '__main__':
    # from  models_tlx.tlx_inceptionv3 import inception_v3

    # model = inception_v3(pretrained=False)
    # Train = ModelTrainTLX(model)
    # Train.train()
    args = parse_args()
    if args.model_name == "alexnet":
    # model = vgg16(pretrained=True, num_classes=2)  # need to freeze network parameters and modify classifier output
        model = tlx_alexnet.alexnet(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "cspdarknet53":
        model = tlx_cspdarknet.cspdarknet53(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "densenet":
        model = tlx_densenet.densenet121(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "dla34":
        model = tlx_dla.dla34(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "dla102":
        model = tlx_dla.dla102(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "dpn68":
        model = tlx_dpn.dpn68(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "dpn107":
        model = tlx_dpn.dpn107(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "efficientnetb1":
        model = tlx_efficientnet.efficientnetb1(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "efficientnetb7":
        model = tlx_efficientnet.efficientnetb7(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "ghostnet":
        model = tlx_ghostnet.ghostnet(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "googlenet":
        model = tlx_googlenet.googlenet(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "hardnet39":
        model = tlx_hardnet.hardnet39(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "hardnet85":
        model = tlx_hardnet.hardnet85(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "inceptionv3":
        model = tlx_inceptionv3.inception_v3(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "mobilenetv1":
        model = tlx_mobilenetv1.mobilenet_v1(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "mobilenetv2":
        model = tlx_mobilenetv2.mobilenet_v2(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "mobilenetv3":
        model = tlx_mobilenetv3.mobilenet_v3_small(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "RedNet50":
        model = tlx_rednet.RedNet50(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "RedNet101":
        model = tlx_rednet.RedNet101(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "vgg16":
        model = tlx_vgg.vgg16(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "resnet50":
        model = tlx_resnet.resnet50(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "resnet101":
        model = tlx_resnet.resnet101(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "ResNeSt50":
        model = tlx_resnest.ResNeSt50(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "ResNeXt50":
        model = tlx_resnext.ResNeXt50_32x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "ResNeXt101":
        model = tlx_resnext.ResNeXt101_64x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "rexnet":
        model = tlx_rexnet.rexnet(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "SE_ResNeXt50":
        model = tlx_se_resnext.SE_ResNeXt50_32x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "SE_ResNeXt101":
        model = tlx_se_resnext.SE_ResNeXt101_32x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "shufflenet":
        model = tlx_shufflenetv2.shufflenet_v2_x0_25(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "squeezenet":
        model = tlx_squeezenet.squeezenet1_0(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "darknet53":
        model = tlx_darknet53.darknet53(pretrained=False)#, num_classes=CLASS_NUM)
    else:
        raise(f"input model name {args.model_name} error")

    Train = ModelTrainTLX(model)
    Train.train(model_name=args.model_name)
