# coding: utf-8
import sys
import importlib


class PaddleClassificationModel(object):
    def __init__(self, project_path, model_name="vgg16"):
        self.model_name = model_name
        self.pd_model = self.load_pd_model(project_path, model_name)

    def load_pd_model(self, pd_project_path, model_name="vgg16"):
        sys.path.insert(0, pd_project_path)
        model = None
        if model_name == "vgg16":
            # from vgg import vgg16 as pd_vgg16
            # model = pd_vgg16(pretrained=True)
            import vgg
            importlib.reload(vgg)
            model = vgg.vgg16(pretrained=True)
        elif model_name == "alexnet":
            # from alexnet import alexnet as pd_alexnet
            # model = pd_alexnet(pretrained=True)
            import alexnet
            importlib.reload(alexnet)
            model = alexnet.alexnet(pretrained=True)
        elif model_name == "resnet50":
            # from resnet import resnet50 as pd_resnet50
            # model = pd_resnet50(pretrained=True)
            import resnet
            importlib.reload(resnet)
            model = resnet.resnet50(pretrained=True)
        elif model_name == "resnet101":
            # from resnet import resnet101 as pd_resnet101
            # model = pd_resnet101(pretrained=True)
            import resnet
            importlib.reload(resnet)
            model = resnet.resnet101(pretrained=True)
        elif model_name == "googlenet":
            # from googlenet import googlenet as pd_googlenet
            # model = pd_googlenet(pretrained=True)
            import googlenet
            importlib.reload(googlenet)
            model = googlenet.googlenet(pretrained=True)
        elif model_name == "mobilenetv1":
            # from mobilenetv1 import mobilenet_v1 as pd_mobilenet_v1
            # model = pd_mobilenet_v1(pretrained=True)
            import mobilenetv1
            importlib.reload(mobilenetv1)
            model = mobilenetv1.mobilenet_v1(pretrained=True)
        elif model_name == "mobilenetv2":
            # from mobilenetv2 import mobilenet_v2 as pd_mobilenet_v2
            # model = pd_mobilenet_v2(pretrained=True)
            import mobilenetv2
            importlib.reload(mobilenetv2)
            model = mobilenetv2.mobilenet_v2(pretrained=True)
        elif model_name == "mobilenetv3":
            # from mobilenetv3 import mobilenet_v3_small as pd_mobilenet_v3_small
            # model = pd_mobilenet_v3_small(pretrained=True)
            import mobilenetv3
            importlib.reload(mobilenetv3)
            model = mobilenetv3.mobilenet_v3_small(pretrained=True)
        elif model_name == "shufflenetv2":
            # from shufflenetv2 import shufflenet_v2_swish as pd_shufflenetv2
            # model = pd_shufflenetv2(pretrained=True)
            import shufflenetv2
            importlib.reload(shufflenetv2)
            # model = shufflenetv2.shufflenet_v2_swish(pretrained=True)
            model = shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        elif model_name == "squeezenet":
            # from squeezenet import squeezenet1_0 as pd_squeezenet1_0
            # model = pd_squeezenet1_0(pretrained=True)
            import squeezenet
            importlib.reload(squeezenet)
            model = squeezenet.squeezenet1_0(pretrained=True)
        elif model_name == "inceptionv3":
            # from inceptionv3 import inception_v3 as pd_inception_v3
            # model = pd_inception_v3(pretrained=True)
            import inceptionv3
            importlib.reload(inceptionv3)
            model = inceptionv3.inception_v3(pretrained=True)
        elif model_name == "regnet":
            # from regnet import regnetx_4gf as pd_regnetx_4gf
            # model = pd_regnetx_4gf(pretrained=True)
            import regnet
            importlib.reload(regnet)
            model = regnet.regnetx_4gf(pretrained=True)
        elif model_name == "tnt":
            # from tnt import tnt_small as pd_tnt_small
            # model = pd_tnt_small(pretrained=True)
            import tnt
            importlib.reload(tnt)
            model = tnt.tnt_small(pretrained=True)
        elif model_name == "darknet53":
            # from darknet53 import darknet53 as pd_darknet53
            # model = pd_darknet53(pretrained=True)
            import darknet53
            importlib.reload(darknet53)
            model = darknet53.darknet53(pretrained=True)
        elif model_name == "densenet":
            # from densenet import densenet121 as pd_densenet121
            # model = pd_densenet121(pretrained=True)
            import densenet
            importlib.reload(densenet)
            model = densenet.densenet121(pretrained=True)
        elif model_name == "rednet50":
            # from rednet import rednet50 as tlx_rednet50
            # model = tlx_rednet50(pretrained=True)
            import rednet
            importlib.reload(rednet)
            model = rednet.rednet50(pretrained=True)
        elif model_name == "rednet101":
            # from rednet import rednet101 as tlx_rednet101
            # model = tlx_rednet101(pretrained=True)
            import rednet
            importlib.reload(rednet)
            model = rednet.rednet101(pretrained=True)
        elif model_name == "cspdarknet53":
            # from cspdarknet import cspdarknet53 as pd_cspdarknet53
            # model = pd_cspdarknet53(pretrained=True)
            import cspdarknet
            importlib.reload(cspdarknet)
            model = cspdarknet.cspdarknet53(pretrained=True)
        elif model_name == "dla34":
            # from dla import dla34 as pd_dla34
            # model = pd_dla34(pretrained=True)
            import dla
            importlib.reload(dla)
            model = dla.dla34(pretrained=True)
        elif model_name == "dla102":
            # from dla import dla102 as tlx_dla102
            # model = tlx_dla102(pretrained=True)
            import dla
            importlib.reload(dla)
            model = dla.dla102(pretrained=True)
        elif model_name == "dpn68":
            # from dpn import dpn68 as tlx_dpn68
            # model = tlx_dpn68(pretrained=True)
            import dpn
            importlib.reload(dpn)
            model = dpn.dpn68(pretrained=True)
        elif model_name == "dpn107":
            # from dpn import dpn107 as tlx_dpn107
            # model = tlx_dpn107(pretrained=True)
            import dpn
            importlib.reload(dpn)
            model = dpn.dpn107(pretrained=True)
        elif model_name == "efficientnet_b1":
            # from efficientnet import efficientnetb1 as tlx_efficientnetb1
            # model = tlx_efficientnetb1(pretrained=True)
            import efficientnet
            importlib.reload(efficientnet)
            model = efficientnet.efficientnetb1(pretrained=True)
        elif model_name == "efficientnet_b7":
            # from efficientnet import efficientnetb7 as tlx_efficientnetb7
            # model = tlx_efficientnetb7(pretrained=True)
            import efficientnet
            importlib.reload(efficientnet)
            model = efficientnet.efficientnetb7(pretrained=True)
        elif model_name == "ghostnet":
            # from ghostnet import ghostnet_x0_5 as pd_ghostnet_x0_5
            # model = pd_ghostnet_x0_5(pretrained=True)
            import ghostnet
            importlib.reload(ghostnet)
            model = ghostnet.ghostnet_x0_5(pretrained=True)
        elif model_name == "hardnet39":
            # from hardnet import hardnet39 as tlx_hardnet39
            # model = tlx_hardnet39(pretrained=True)
            import hardnet
            importlib.reload(hardnet)
            model = hardnet.hardnet39(pretrained=True)
        elif model_name == "hardnet85":
            # from hardnet import hardnet85 as tlx_hardnet85
            # model = tlx_hardnet85(pretrained=True)
            import hardnet
            importlib.reload(hardnet)
            model = hardnet.hardnet85(pretrained=True)
        elif model_name == "resnest50":
            # from resnest import resnest50 as pd_resnest50
            # model = pd_resnest50(pretrained=True)
            import resnest
            importlib.reload(resnest)
            model = resnest.resnest50(pretrained=True)
        elif model_name == "resnext50":
            # from resnext import resnext50_32x4d as pd_resnext50_32x4d
            # model = pd_resnext50_32x4d(pretrained=True)
            import resnext
            importlib.reload(resnext)
            model = resnext.resnext50_32x4d(pretrained=True)
        elif model_name == "resnext101":
            # from resnext import resnext101_32x4d as tlx_resnext101_32x4d
            # model = tlx_resnext101_32x4d(pretrained=True)
            import resnext
            importlib.reload(resnext)
            model = resnext.resnext101_32x4d(pretrained=True)
        elif model_name == "rexnet":
            # from rexnet import rexnet_1_0 as pd_rexnet_1_0
            # model = pd_rexnet_1_0(pretrained=True)
            import rexnet
            importlib.reload(rexnet)
            model = rexnet.rexnet_1_0(pretrained=True)
        elif model_name == "se_resnext":
            # from se_resnext import se_resnext50_32x4d as pd_se_resnext50_32x4d
            # model = pd_se_resnext50_32x4d(pretrained=True)
            import se_resnext
            importlib.reload(se_resnext)
            model = se_resnext.se_resnext50_32x4d(pretrained=True)
        elif model_name == "esnet_x0_5":
            # from esnet import esnet_x0_5 as tlx_esnet_x0_5
            # model = tlx_esnet_x0_5(pretrained=True)
            import esnet
            importlib.reload(esnet)
            model = esnet.esnet_x0_5(pretrained=True)
        elif model_name == "esnet_x1_0":
            # from esnet import esnet_x1_0 as tlx_esnet_x1_0
            # model = tlx_esnet_x1_0(pretrained=True)
            import esnet
            importlib.reload(esnet)
            model = esnet.esnet_x1_0(pretrained=True)
        elif model_name == "vit":
            # from vision_transformer import vit_small_patch16_224 as pd_vit_small_patch16_224
            # model = pd_vit_small_patch16_224(pretrained=True)
            import vision_transformer
            importlib.reload(vision_transformer)
            model = vision_transformer.vit_small_patch16_224(pretrained=True)
        elif model_name == "alt_gvt_small":
            # from gvt import alt_gvt_small as tlx_alt_gvt_small
            # model = tlx_alt_gvt_small(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.alt_gvt_small(pretrained=True)
        elif model_name == "alt_gvt_base":
            # from gvt import alt_gvt_base as tlx_alt_gvt_base
            # model = tlx_alt_gvt_base(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.alt_gvt_base(pretrained=True)
        elif model_name == "pcpvt_base":
            # from gvt import pcpvt_base as tlx_pcpvt_base
            # model = tlx_pcpvt_base(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.pcpvt_base(pretrained=True)
        elif model_name == "pcpvt_large":
            # from gvt import pcpvt_large as tlx_pcpvt_large
            # model = tlx_ppcpvt_large(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.pcpvt_large(pretrained=True)
        elif model_name == "swin_transformer_base":
            # from swin_transformer import swintransformer_base_patch4_window7_224 as tlx_swintransformer_base
            # model = tlx_swintransformer_base(pretrained=True)
            import swin_transformer
            importlib.reload(swin_transformer)
            model = swin_transformer.swintransformer_base_patch4_window7_224(pretrained=True)
        elif model_name == "swin_transformer_small":
            # from swin_transformer import swintransformer_small_patch4_window7_224 as pd_swintransformer_small
            # model = pd_swintransformer_small(pretrained=True)
            import swin_transformer
            importlib.reload(swin_transformer)
            model = swin_transformer.swintransformer_small_patch4_window7_224(pretrained=True)
        elif model_name == "xception41":
            # from xception import xception41 as pd_xception41
            # model = pd_xception41(pretrained=True)
            import xception
            importlib.reload(xception)
            model = xception.xception41(pretrained=True)
        elif model_name == "xception65":
            # from xception import xception65 as tlx_xception65
            # model = tlx_xception65(pretrained=True)
            import xception
            importlib.reload(xception)
            model = xception.xception65(pretrained=True)
        elif model_name == "xception41_deeplab":
            # from xception_deeplab import xception41_deeplab as tlx_xception41_deeplab
            # model = tlx_xception41_deeplab(pretrained=True)
            import xception_deeplab
            importlib.reload(xception_deeplab)
            model = xception_deeplab.xception41_deeplab(pretrained=True)
        elif model_name == "xception65_deeplab":
            # from xception_deeplab import xception65_deeplab as tlx_xception65_deeplab
            # model = tlx_xception65_deeplab(pretrained=True)
            import xception_deeplab
            importlib.reload(xception_deeplab)
            model = xception_deeplab.xception65_deeplab(pretrained=True)
        elif model_name == "levit":
            # from levit import levit_128 as pd_levit_128
            # model = pd_levit_128(pretrained=True)
            import levit
            importlib.reload(levit)
            model = levit.levit_128(pretrained=True)
        elif model_name == "mixnet":
            # from mixnet import mixnet_s as pd_mixnet_s
            # model = pd_mixnet_s(pretrained=True)
            import mixnet
            importlib.reload(mixnet)
            model = mixnet.mixnet_s(pretrained=True)
        # todo - new added model
        elif model_name == "convnext":  # 序号103 nn.Identity转换失败
            import convnext
            importlib.reload(convnext)
            model = convnext.convnext(pretrained=True)
        elif model_name == "cswin":  # 序号101 tlx.ops.gelu不该加()
            import cswin_transformer
            importlib.reload(cswin_transformer)
            model = cswin_transformer.CSwintransformer_thiny(pretrained=True)
        elif model_name == "deittiny":  # 序号49
            import deit
            importlib.reload(deit)
            model = deit.DeiT_tiny(pretrained=True)
        elif model_name == "deitsmall":  # 序号50
            import deit
            importlib.reload(deit)
            model = deit.DeiT_small(pretrained=True)
        elif model_name == "deitbase":  # 序号51
            import deit
            importlib.reload(deit)
            model = deit.DeiT_base(pretrained=True)
        elif model_name == "dvt":  # 序号6
            import distilled_vision_transformer
            importlib.reload(distilled_vision_transformer)
            model = distilled_vision_transformer.Distilled_vision_transformer_base(pretrained=True)
        elif model_name == "peleenet":  # 序号102
            import peleenet
            importlib.reload(peleenet)
            model = peleenet.peleenet(pretrained=True)
        elif model_name == "pp_hgnet":  # 序号106
            import pp_hgnet
            importlib.reload(pp_hgnet)
            model = pp_hgnet.pp_hgnet(pretrained=True)
        elif model_name == "pp_lcnet":  # 序号107
            import pp_lcnet
            importlib.reload(pp_lcnet)
            model = pp_lcnet.pp_lcnet(pretrained=True)
        elif model_name == "pp_lcnet_v2":  # 序号108
            import pp_lcnet_v2
            importlib.reload(pp_lcnet_v2)
            model = pp_lcnet_v2.pp_lcnetv2(pretrained=True)
        elif model_name == "pvt_v2":  # 序号109
            import pvt_v2
            importlib.reload(pvt_v2)
            model = pvt_v2.pvt_v2(pretrained=True)
        elif model_name == "res2net":  # 序号110
            import res2net
            importlib.reload(res2net)
            model = res2net.res2net(pretrained=True)
        elif model_name == "van":  # 序号104
            import van
            importlib.reload(van)
            model = van.van(pretrained=True)
        # sys.path.remove(pd_project_path)
        return model
