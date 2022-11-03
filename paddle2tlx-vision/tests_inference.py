# coding: utf-8
import warnings
warnings.filterwarnings("ignore")
import unittest
import numpy as np
from predict_vision import predict_pd, predict_tlx


class PaddleModelTest(unittest.TestCase):
    def test_lenet(self):
        import paddle
        from models.vision.pd_lenet import LeNet

        model = LeNet()
        model.eval()
        img = paddle.rand([1, 1, 28, 28])
        out = model(img)
        print(np.argmax(out[0]))

    def test_vgg(self):
        from models_pd.pd_vgg import vgg11, vgg13, vgg16, vgg19

        image_file = "images/dog.jpeg"
        # model = vgg11(pretrained=False, batch_norm=False)  # vgg11 model do not have a pretrained model now
        # model.eval()
        # predict_pd(model, image_file)

        # model = vgg13(pretrained=False, batch_norm=False)  # vgg13 model do not have a pretrained model now
        # model.eval()
        # predict_pd(model, image_file)

        model = vgg16(pretrained=True, batch_norm=False)
        # model = vgg16(batch_norm=True)
        model.eval()
        predict_pd(model, image_file)

        # model = vgg19(pretrained=True, batch_norm=False)
        # model.eval()
        # predict_pd(model, image_file)

    def test_alexnet(self):
        from models_pd.pd_alexnet import alexnet

        image_file = "images/dog.jpeg"
        model = alexnet(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

    def test_googlenet(self):
        from models_pd.pd_googlenet import googlenet

        image_file = "images/dog.jpeg"
        model = googlenet(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

    def test_squeezenet(self):
        from models_pd.pd_squeezenet import squeezenet1_0, squeezenet1_1

        image_file = "images/dog.jpeg"
        model = squeezenet1_0(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        model = squeezenet1_1(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

    def test_resnet(self):
        from models_pd.pd_resnet import resnet18, resnet34, resnet50, resnet101, resnet152

        image_file = "images/dog.jpeg"
        # model = resnet18(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)

        # model = resnet34(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)

        model = resnet50(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        # model = resnet101(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)
        #
        # model = resnet152(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)

    def test_densenet(self):
        from models_pd.pd_densenet import densenet121, densenet161, densenet169, densenet201, densenet264

        image_file = "images/dog.jpeg"
        model = densenet121(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        # model = densenet161(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)
        #
        # model = densenet169(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)
        #
        # model = densenet201(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)
        #
        # model = densenet264(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)

    def test_mobilenet(self):
        from models_pd.pd_mobilenetv1 import mobilenet_v1
        from models_pd.pd_mobilenetv2 import mobilenet_v2
        from models_pd.pd_mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large

        image_file = "images/dog.jpeg"
        model = mobilenet_v1(pretrained=True)
        import paddle
        print(paddle.summary(model, (1, 3, 224, 224)))
        model.eval()
        predict_pd(model, image_file)

        # model = mobilenet_v2(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)
        #
        # model = mobilenet_v3_small(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)
        #
        # model = mobilenet_v3_large(pretrained=True)
        # model.eval()
        # predict_pd(model, image_file)

    def test_inception(self):
        from models_pd.pd_inceptionv3 import inception_v3

        image_file = "images/dog.jpeg"
        model = inception_v3(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

    def test_shufflenet(self):
        from models_pd.pd_shufflenetv2 import shufflenet_v2_x0_25, shufflenet_v2_x0_33, shufflenet_v2_x0_5
        from models_pd.pd_shufflenetv2 import shufflenet_v2_x1_0, shufflenet_v2_x1_5
        from models_pd.pd_shufflenetv2 import shufflenet_v2_x2_0, shufflenet_v2_swish

        image_file = "images/dog.jpeg"
        model = shufflenet_v2_x0_25(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        model = shufflenet_v2_x0_33(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        model = shufflenet_v2_x0_5(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        model = shufflenet_v2_x1_0(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        model = shufflenet_v2_x1_5(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        model = shufflenet_v2_x2_0(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

        model = shufflenet_v2_swish(pretrained=True)
        model.eval()
        predict_pd(model, image_file)

    def test_darknet53(self):
        from models_pd.pd_darknet53 import darknet53

        image_file = "images/dog.jpeg"
        model = darknet53(pretrained=True)
        model.eval()
        predict_pd(model, image_file)


class TLXModelTest(unittest.TestCase):
    # pass
    def test_lenet(self):
        import os
        os.environ['TL_BACKEND'] = 'paddle'  # config paddle as backend in first
        import tensorlayerx as tlx
        from models_tlx.tlx_lenet import LeNet

        model = LeNet()
        # print(model)
        img = np.random.random(size=(1, 1, 28, 28)).astype('float32')
        img = tlx.convert_to_tensor(img)
        out = model(img)
        print(np.argmax(out[0]))

    def test_vgg(self):
        from models_tlx.tlx_vgg import vgg11, vgg13, vgg16, vgg19

        image_file = "images/dog.jpeg"
        # model = vgg11(pretrained=False, batch_norm=False)  # vgg11 model do not have a pretrained model now
        # model.set_eval()
        # predict_tlx(model, image_file)
        #
        # model = vgg13(pretrained=False, batch_norm=False)  # vgg13 model do not have a pretrained model now
        # model.set_eval()
        # predict_tlx(model, image_file)

        model = vgg16(pretrained=True, batch_norm=False)
        # model = vgg16(batch_norm=True)  # TODO vgg16_bn model do not have a pretrained model now
        model.set_eval()
        predict_tlx(model, image_file)

        # model = vgg19(pretrained=True, batch_norm=False)
        # model.set_eval()
        # predict_tlx(model, image_file)

    def test_alexnet(self):
        from models_tlx.tlx_alexnet import alexnet

        image_file = "images/dog.jpeg"
        model = alexnet(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

    def test_googlenet(self):
        from models_tlx.tlx_googlenet import googlenet

        image_file = "images/dog.jpeg"
        model = googlenet(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

    def test_squeezenet(self):
        from models_tlx.tlx_squeezenet import squeezenet1_0, squeezenet1_1

        image_file = "images/dog.jpeg"
        model = squeezenet1_0(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        model = squeezenet1_1(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

    # diff is large
    def test_resnet(self):
        from models_tlx.tlx_resnet import resnet18, resnet34, resnet50, resnet101, resnet152

        image_file = "images/dog.jpeg"
        # model = resnet18(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)

        # model = resnet34(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)

        model = resnet50(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        # model = resnet101(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)
        #
        # model = resnet152(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)

    def test_densenet(self):
        from models_tlx.tlx_densenet import densenet121, densenet161, densenet169, densenet201, densenet264

        image_file = "images/dog.jpeg"
        model = densenet121(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        # model = densenet161(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)
        #
        # model = densenet169(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)
        #
        # model = densenet201(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)
        #
        # model = densenet264(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)

    # not pass
    def test_mobilenet(self):
        from models_tlx.tlx_mobilenetv1 import mobilenet_v1
        from models_tlx.tlx_mobilenetv2 import mobilenet_v2
        from models_tlx.tlx_mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large

        image_file = "images/dog.jpeg"
        model = mobilenet_v1(pretrained=True)
        # import paddle
        # print(paddle.summary(model, (1, 3, 224, 224)))
        model.set_eval()
        predict_tlx(model, image_file)

        model = mobilenet_v2(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        # model = mobilenet_v3_small(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)
        #
        # model = mobilenet_v3_large(pretrained=True)
        # model.set_eval()
        # predict_tlx(model, image_file)

    def test_inception(self):
        from models_tlx.tlx_inceptionv3 import inception_v3

        image_file = "images/dog.jpeg"
        model = inception_v3(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

    def test_shufflenet(self):
        from models_tlx.tlx_shufflenetv2 import shufflenet_v2_x0_25, shufflenet_v2_x0_33, shufflenet_v2_x0_5
        from models_tlx.tlx_shufflenetv2 import shufflenet_v2_x1_0, shufflenet_v2_x1_5
        from models_tlx.tlx_shufflenetv2 import shufflenet_v2_x2_0, shufflenet_v2_swish

        image_file = "images/dog.jpeg"
        model = shufflenet_v2_x0_25(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        model = shufflenet_v2_x0_33(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        model = shufflenet_v2_x0_5(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        model = shufflenet_v2_x1_0(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        model = shufflenet_v2_x1_5(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        model = shufflenet_v2_x2_0(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

        model = shufflenet_v2_swish(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)

    def test_darknet53(self):
        from models_tlx.tlx_darknet53 import darknet53

        image_file = "images/dog.jpeg"
        model = darknet53(pretrained=True)
        model.set_eval()
        predict_tlx(model, image_file)


if __name__ == '__main__':
    unittest.main()
