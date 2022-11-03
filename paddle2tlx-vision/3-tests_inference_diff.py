# coding: utf-8
import os

os.environ['TL_BACKEND'] = 'paddle'

import unittest
from predict_vision import calc_diff
# import sys
# sys.path.append("/home/sthq/zhuxc/paddle2tlx-vision/")
image_file = "images/dog.jpeg"

class InferenceModelDiffTest(unittest.TestCase):
    # ok,droupout问题及pooling的padding默认值问题
    def test_alexnet_1(self):
        print("alexnet begin.....")
        from models_pd import pd_alexnet
        from models_tlx import tlx_alexnet

        model_tlx = tlx_alexnet.alexnet(pretrained=True)
        model_pd = pd_alexnet.alexnet(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="alexnet")
        print("alexnet end.....")

    # ok
    def test_cspdarknet_2(self):
        """diff 0"""
        print("cspdarknet53 begin.....")
        from models_tlx import tlx_cspdarknet
        from models_pd import pd_cspdarknet
        model_tlx = tlx_cspdarknet.cspdarknet53(pretrained=True)
        model_pd = pd_cspdarknet.cspdarknet53(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file, mode_name="cspdarknet53")
        print("cspdarknet53 end.....")

    # ok
    def test_densenet_5(self):
        from models_pd import pd_densenet
        from models_tlx import tlx_densenet
        model_tlx = tlx_densenet.densenet121(pretrained=True)
        model_pd = pd_densenet.densenet121(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="densenet121")

        # model_tlx = tlx_densenet.densenet161(pretrained=True)
        # model_pd = pd_densenet.densenet161(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="densenet161")

        # model_tlx = tlx_densenet.densenet169(pretrained=True)
        # model_pd = pd_densenet.densenet169(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="densenet169")

        # model_tlx = tlx_densenet.densenet201(pretrained=True)
        # model_pd = pd_densenet.densenet201(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="densenet201")

        # model_tlx = tlx_densenet.densenet264(pretrained=True)
        # model_pd = pd_densenet.densenet264(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="densenet264")

    #######dla:ok########
    def test_dla_7_8(self):
        print("dla begin.....")
        from models_pd import pd_dla
        from models_tlx import tlx_dla
        # dla34
        model_tlx = tlx_dla.dla34(pretrained=True)
        model_pd = pd_dla.dla34(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="dla34")
        # dla102
        model_pd = pd_dla.dla102(pretrained=True)
        model_tlx = tlx_dla.dla102(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="dla102")
        print("dla end.....")

    # ok
    def test_dnp_9_10(self):
        """diff 0"""
        from models_tlx import tlx_dpn
        from models_pd import pd_dpn
        model_tlx = tlx_dpn.dpn68(pretrained=True)
        model_pd = pd_dpn.dpn68(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)
    # ok
    def test_dnp107(self):
        """diff 0"""
        from models_tlx import tlx_dpn
        from models_pd import pd_dpn
        model_tlx = tlx_dpn.dpn107(pretrained=True)
        model_pd = pd_dpn.dpn107(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file, mode_name="dpn107")
    # ok
    def test_efficientnet_11_12(self):
        # 参数没完全对上
        print("efficientnetb1 begin.....")
        from models_pd import pd_efficientnet
        from models_tlx import tlx_efficientnet

        model_tlx = tlx_efficientnet.efficientnetb1(pretrained=True)
        model_pd = pd_efficientnet.efficientnetb1(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="efficientnetb1")
        print("efficientnetb1 end.....")

    # ok
    def test_efficientnet_11_12(self):
        from models_pd import pd_efficientnet
        from models_tlx import tlx_efficientnet
        model_tlx = tlx_efficientnet.efficientnetb7(pretrained=True)
        model_pd = pd_efficientnet.efficientnetb7(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="efficientnetb7")
        print("efficientnetb1 end.....")
    # ok
    def test_ghostnet_13(self):
        from models_pd import pd_ghostnet
        from models_tlx import tlx_ghostnet
        model_tlx = tlx_ghostnet.ghostnet(pretrained=True)
        model_pd = pd_ghostnet.ghostnet(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="ghostnet")
    # ok
    def test_googlenet_14(self):
        from models_pd import pd_googlenet
        from models_tlx import tlx_googlenet

        model_tlx = tlx_googlenet.googlenet(pretrained=True)
        model_pd = pd_googlenet.googlenet(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="googlenet")

    # 修正groupconv后ok
    def test_hardnet_17_18(self):
        """diff 0"""
        from models_tlx import tlx_hardnet
        from models_pd import pd_hardnet
        model_tlx = tlx_hardnet.hardnet39(pretrained=True)
        model_pd = pd_hardnet.hardnet39(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file, mode_name="hardnet39")

        model_tlx = tlx_hardnet.hardnet85(pretrained=True)
        model_pd = pd_hardnet.hardnet85(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="hardnet85")

    #  ok: AvgPool2D算子的exclusive参数，True时ok,但False转时出错
    def test_inception_19(self):
        from models_pd import pd_inceptionv3
        from models_tlx import tlx_inceptionv3
        model_tlx = tlx_inceptionv3.inception_v3(pretrained=True)
        model_pd = pd_inceptionv3.inception_v3(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="inceptionv3")


    # ok
    def test_mobilenet_23_24_25(self):
        from models_pd import pd_mobilenetv1, pd_mobilenetv2, pd_mobilenetv3
        from models_tlx import tlx_mobilenetv1, tlx_mobilenetv2, tlx_mobilenetv3
        model_tlx = tlx_mobilenetv1.mobilenet_v1(pretrained=True)
        model_pd = pd_mobilenetv1.mobilenet_v1(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="MobilenetV1")

        model_tlx = tlx_mobilenetv2.mobilenet_v2(pretrained=True)
        model_pd = pd_mobilenetv2.mobilenet_v2(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="MobilenetV2")
        # 不一致：模型参数修改后ok
        model_pd = pd_mobilenetv3.mobilenet_v3_small(pretrained=True)
        model_tlx = tlx_mobilenetv3.mobilenet_v3_small(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="MobilenetV3_small")

        model_tlx = tlx_mobilenetv3.mobilenet_v3_large(pretrained=True)
        model_pd = pd_mobilenetv3.mobilenet_v3_large(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="mobilenet_v3_large")
    # ok
    def test_rednet_28_29(self):
        from models_pd import pd_rednet
        from models_tlx import tlx_rednet

        model_tlx = tlx_rednet.RedNet50(pretrained=True)
        model_pd = pd_rednet.RedNet50(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="RedNet50")

        model_tlx = tlx_rednet.RedNet101(pretrained=True)
        model_pd = pd_rednet.RedNet101(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="RedNet101")

    # ok
    def test_vgg_31(self):
        from models_pd import pd_vgg
        from models_tlx import tlx_vgg

        # model_tlx = tlx_vgg.vgg11(pretrained=False)
        # model_pd = pd_vgg.vgg11(pretrained=False)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="vgg11")

        # model_tlx = tlx_vgg.vgg13(pretrained=False)
        # model_pd = pd_vgg.vgg13(pretrained=False)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="vgg13")

        model_tlx = tlx_vgg.vgg16(pretrained=True)
        model_pd = pd_vgg.vgg16(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="VGG16")

        # model_tlx = tlx_vgg.vgg19(pretrained=True)
        # model_pd = pd_vgg.vgg19(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="vgg19")

    # ok
    def test_resnet_32_33(self):
        from models_pd import pd_resnet
        from models_tlx import tlx_resnet
        # model_tlx = tlx_resnet.resnet18(pretrained=True)
        # model_pd = pd_resnet.resnet18(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="ResNet18")

        # model_tlx = tlx_resnet.resnet34(pretrained=True)
        # model_pd = pd_resnet.resnet34(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="ResNet34")

        model_tlx = tlx_resnet.resnet50(pretrained=True)
        model_pd = pd_resnet.resnet50(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="ResNet50")

        model_tlx = tlx_resnet.resnet101(pretrained=True)
        model_pd = pd_resnet.resnet101(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="ResNet101")

        # model_tlx = tlx_resnet.resnet152(pretrained=True)
        # model_pd = pd_resnet.resnet152(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="ResNet152")
    # ok
    def test_resnest_34(self):
        print("ResNeSt begin.....")
        from models_pd import pd_resnest
        from models_tlx import tlx_resnest
        model_tlx = tlx_resnest.ResNeSt50(pretrained=True)
        model_pd = pd_resnest.ResNeSt50(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="ResNeSt")
        print("ResNeSt end.....")

    # ok
    def test_resnext50_35(self):
        print("ResNeXt50 begin.....")
        from models_pd import pd_resnext
        from models_tlx import tlx_resnext
        model_tlx = tlx_resnext.ResNeXt50_32x4d(pretrained=True)
        model_pd = pd_resnext.ResNeXt50_32x4d(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="ResNeXt50")
        print("ResNeXt50 begin.....")

    # ok
    def test_resnext101_36(self):
        print("ResNeXt101 begin.....")
        from models_pd import pd_resnext
        from models_tlx import tlx_resnext
        model_tlx = tlx_resnext.ResNeXt101_64x4d(pretrained=True)
        model_pd = pd_resnext.ResNeXt101_64x4d(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="ResNeXt101")
        print("ResNeXt101 end.....")

    Model ReXNetV1 predict category - TLX: Great Pyrenees
    Model ReXNetV1 predict category - Paddle: Great Pyrenees
    diff sum value: 0.2349165
    diff max value: 0.0012648106
    def test_rexnet_37(self):
        print("rexnet begin.....")
        from models_pd import pd_rexnet
        from models_tlx import tlx_rexnet
        model_tlx = tlx_rexnet.rexnet(pretrained=True)
        model_pd = pd_rexnet.rexnet(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="rexnet")
        print("rexnet end.....")
    # ok
    def test_se_resnext50_38(self):
        from models_pd import pd_se_resnext
        from models_tlx import tlx_se_resnext
        model_tlx = tlx_se_resnext.SE_ResNeXt50_32x4d(pretrained=True)
        model_pd = pd_se_resnext.SE_ResNeXt50_32x4d(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="SE_ResNeXt50")
    # ok
    def test_se_resnext101_38(self):
        from models_pd import pd_se_resnext
        from models_tlx import tlx_se_resnext
        model_tlx = tlx_se_resnext.SE_ResNeXt101_32x4d(pretrained=True)
        model_pd = pd_se_resnext.SE_ResNeXt101_32x4d(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="SE_ResNeXt101_32x4d")
    # ok
    def test_shufflenet_39(self):
        from models_pd import pd_shufflenetv2
        from models_tlx import tlx_shufflenetv2
        model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="ShuffleNet_v2")

        # model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_33(pretrained=True)
        # model_pd = pd_shufflenetv2.shufflenet_v2_x0_33(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="shufflenet_v2_x0_33")

        # model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
        # model_pd = pd_shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="shufflenet_v2_x0_5")

        # model_tlx = tlx_shufflenetv2.shufflenet_v2_x1_0(pretrained=True)
        # model_pd = pd_shufflenetv2.shufflenet_v2_x1_0(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="shufflenet_v2_x1_0")

        # model_tlx = tlx_shufflenetv2.shufflenet_v2_x1_5(pretrained=True)
        # model_pd = pd_shufflenetv2.shufflenet_v2_x1_5(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="shufflenet_v2_x1_5")

        # model_tlx = tlx_shufflenetv2.shufflenet_v2_x2_0(pretrained=True)
        # model_pd = pd_shufflenetv2.shufflenet_v2_x2_0(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="shufflenet_v2_x2_0")

        # model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        # model_pd = pd_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="shufflenet_v2_x0_25")

    # ok, dropout问题
    def test_squeezenet_40(self):
        from models_pd import pd_squeezenet
        from models_tlx import tlx_squeezenet
        model_tlx = tlx_squeezenet.squeezenet1_0(pretrained=True)
        model_pd = pd_squeezenet.squeezenet1_0(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="SqueezeNet")

        # model_tlx = tlx_squeezenet.squeezenet1_1(pretrained=True)
        # model_pd = pd_squeezenet.squeezenet1_1(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file,mode_name="squeezenet1_1")

    # ok
    def test_darknet_46(self):
        from models_pd import pd_darknet53
        from models_tlx import tlx_darknet53
        model_tlx = tlx_darknet53.darknet53(pretrained=True)
        model_pd = pd_darknet53.darknet53(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file,mode_name="Darknet53")

if __name__ == '__main__':
    unittest.main()
