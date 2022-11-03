# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import paddle
import numpy as np
from utils.options import parse_args
from models_pd import pd_alexnet
from models_pd import pd_cspdarknet
from models_pd import pd_densenet
from models_pd import pd_dla
from models_pd import pd_dpn
from models_pd import pd_efficientnet
from models_pd import pd_ghostnet
from models_pd import pd_googlenet
from models_pd import pd_hardnet
from models_pd import pd_inceptionv3
from models_pd import pd_mobilenetv1, pd_mobilenetv2, pd_mobilenetv3
from models_pd import pd_rednet
from models_pd import pd_vgg
from models_pd import pd_resnet
from models_pd import pd_resnest
from models_pd import pd_resnext
from models_pd import pd_resnext
from models_pd import pd_rexnet
from models_pd import pd_se_resnext
from models_pd import pd_shufflenetv2
from models_pd import pd_squeezenet
from models_pd import pd_darknet53

# EPOCH_NUM = 10
EPOCH_NUM = 4
BATCH_SIZE = 8  # 64
BATCH_NUM = 4  # 100
IMAGE_SHAPE = [3, 224, 224]
CLASS_NUM = 1000

class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random(size=IMAGE_SHAPE).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class ModelTrainPaddle(object):
    def __init__(self, model):
        self.model = model

    def train(self):
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
            # for batch_id, (image, label) in enumerate(loader()):
            for batch_id, (image, label) in enumerate(loader):  # TODO
                out = model(image)
                if isinstance(out,list):
                    out = out[0]
                loss = loss_fn(out, label)
                loss.backward()
                adam.step()
                adam.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(epoch_id, batch_id, np.mean(loss.numpy())))
                # for key in self.model.state_dict().keys():
                #     print(key, self.model.state_dict()[key].shape)
                # for key in self.model.state_dict().keys():
                #     print(key, self.model.state_dict()[key].shape)
                exit()
        print(f"{args.model_name} end...")


if __name__ == '__main__':
    args = parse_args()
    if args.model_name == "alexnet":
    # model = vgg16(pretrained=True, num_classes=2)  # need to freeze network parameters and modify classifier output
        model = pd_alexnet.alexnet(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "cspdarknet53":
        model = pd_cspdarknet.cspdarknet53(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "densenet":
        model = pd_densenet.densenet121(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "dla34":
        model = pd_dla.dla34(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "dla102":
        model = pd_dla.dla102(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "dpn68":
        model = pd_dpn.dpn68(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "dpn107":
        model = pd_dpn.dpn107(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "efficientnetb1":
        model = pd_efficientnet.efficientnetb1(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "efficientnetb7":
        model = pd_efficientnet.efficientnetb7(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "ghostnet":
        model = pd_ghostnet.ghostnet(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "googlenet":
        model = pd_googlenet.googlenet(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "hardnet39":
        model = pd_hardnet.hardnet39(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "hardnet85":
        model = pd_hardnet.hardnet85(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "inceptionv3":
        model = pd_inceptionv3.inception_v3(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "mobilenetv1":
        model = pd_mobilenetv1.mobilenet_v1(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "mobilenetv2":
        model = pd_mobilenetv2.mobilenet_v2(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "mobilenetv3":
        model = pd_mobilenetv3.mobilenet_v3_small(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "RedNet50":
        model = pd_rednet.RedNet50(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "RedNet101":
        model = pd_rednet.RedNet101(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "vgg16":
        model = pd_vgg.vgg16(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "resnet50":
        model = pd_resnet.resnet50(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "resnet101":
        model = pd_resnet.resnet101(pretrained=False, num_classes=CLASS_NUM)
    elif args.model_name == "ResNeSt50":
        model = pd_resnest.ResNeSt50(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "ResNeXt50":
        model = pd_resnext.ResNeXt50_32x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "ResNeXt101":
        model = pd_resnext.ResNeXt101_64x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "rexnet":
        model = pd_rexnet.rexnet(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "SE_ResNeXt50":
        model = pd_se_resnext.SE_ResNeXt50_32x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "SE_ResNeXt101":
        model = pd_se_resnext.SE_ResNeXt101_32x4d(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "shufflenet":
        model = pd_shufflenetv2.shufflenet_v2_x0_25(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "squeezenet":
        model = pd_squeezenet.squeezenet1_0(pretrained=False)#, num_classes=CLASS_NUM)
    elif args.model_name == "darknet53":
        model = pd_darknet53.darknet53(pretrained=False)#, num_classes=CLASS_NUM)
    else:
        raise(f"input model name {args.model_name} error")
    Train = ModelTrainPaddle(model)
    Train.train()
