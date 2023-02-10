# coding: utf-8
import sys
import importlib


class TLXClassificationModel(object):
    def __init__(self, project_path, model_name="vgg16"):
        self.model_name = model_name
        self.tlx_model = self.load_tlx_model(project_path, model_name)

    def load_tlx_model(self, tlx_project_path, model_name="vgg16"):
        import os
        os.environ["TL_BACKEND"] = "paddle"
        sys.path.insert(0, tlx_project_path)
        model = None
        if model_name == "vgg16":  # 序号31 - ok
            # from vgg import vgg16 as tlx_vgg16
            # model = tlx_vgg16(pretrained=True)
            import vgg
            importlib.reload(vgg)
            model = vgg.vgg16(pretrained=True)
        elif model_name == "alexnet":  # 序号1 - ok
            # from alexnet import alexnet as tlx_alexnet
            # model = tlx_alexnet(pretrained=True)
            import alexnet
            importlib.reload(alexnet)
            model = alexnet.alexnet(pretrained=True)
        elif model_name == "resnet50":  # 序号32 - ok
            # from resnet import resnet50 as tlx_resnet50
            # model = tlx_resnet50(pretrained=True)
            import resnet
            importlib.reload(resnet)
            model = resnet.resnet50(pretrained=True)
        elif model_name == "resnet101":  # 序号33 - ok
            # from resnet import resnet101 as tlx_resnet101
            # model = tlx_resnet101(pretrained=True)
            import resnet
            importlib.reload(resnet)
            model = resnet.resnet101(pretrained=True)
        elif model_name == "googlenet":  # 序号14 - ok
            # from googlenet import googlenet as tlx_googlenet
            # model = tlx_googlenet(pretrained=True)
            import googlenet
            importlib.reload(googlenet)
            model = googlenet.googlenet(pretrained=True)
        elif model_name == "mobilenetv1":  # 序号23 - ok - TODO - 转换后单独算误差为0
            # from mobilenetv1 import mobilenet_v1 as tlx_mobilenet_v1
            # model = tlx_mobilenet_v1(pretrained=True)
            import mobilenetv1
            importlib.reload(mobilenetv1)
            model = mobilenetv1.mobilenet_v1(pretrained=True)
        elif model_name == "mobilenetv2":  # 序号24 - ok - TODO - 转换后单独算误差为0
            # from mobilenetv2 import mobilenet_v2 as tlx_mobilenet_v2
            # model = tlx_mobilenet_v2(pretrained=True)
            import mobilenetv2
            importlib.reload(mobilenetv2)
            model = mobilenetv2.mobilenet_v2(pretrained=True)
        elif model_name == "mobilenetv3":  # 序号25 - ok - TODO - 转换后单独算误差为0
            # from mobilenetv3 import mobilenet_v3_small as tlx_mobilenet_v3_small
            # model = tlx_mobilenet_v3_small(pretrained=True)
            import mobilenetv3
            importlib.reload(mobilenetv3)
            model = mobilenetv3.mobilenet_v3_small(pretrained=True)
        elif model_name == "shufflenetv2":  # 序号39 - ok - TODO - 转换后单独算误差为0
            # from shufflenetv2 import shufflenet_v2_swish as tlx_shufflenetv2
            # model = tlx_shufflenetv2(pretrained=True)
            import shufflenetv2
            importlib.reload(shufflenetv2)
            # model = shufflenetv2.shufflenet_v2_swish(pretrained=True)  # diff - 0.0016313102
            model = shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        elif model_name == "squeezenet":  # 序号40 - ok
            # from squeezenet import squeezenet1_0 as tlx_squeezenet1_0
            # model = tlx_squeezenet1_0(pretrained=True)
            import squeezenet
            importlib.reload(squeezenet)
            model = squeezenet.squeezenet1_0(pretrained=True)
        elif model_name == "inceptionv3":  # 序号19 - ok - TODO - 转换后单独算误差为0
            # from inceptionv3 import inception_v3 as tlx_inception_v3
            # model = tlx_inception_v3(pretrained=True)
            import inceptionv3
            importlib.reload(inceptionv3)
            model = inceptionv3.inception_v3(pretrained=True)
        elif model_name == "regnet":  # 序号30 - ok
            # from regnet import regnetx_4gf as tlx_regnetx_4gf
            # model = tlx_regnetx_4gf(pretrained=True)
            import regnet
            importlib.reload(regnet)
            model = regnet.regnetx_4gf(pretrained=True)
        elif model_name == "tnt":  # 序号41 - ok
            # from tnt import tnt_small as tlx_tnt_small
            # model = tlx_tnt_small(pretrained=True)
            import tnt
            importlib.reload(tnt)
            model = tnt.tnt_small(pretrained=True)
        elif model_name == "darknet53":  # 序号46 - ok
            # from darknet53 import darknet53 as tlx_darknet53
            # model = tlx_darknet53(pretrained=True)
            import darknet53
            importlib.reload(darknet53)
            model = darknet53.darknet53(pretrained=True)
        elif model_name == "densenet":  # 序号5 - ok
            # from densenet import densenet121 as tlx_densenet121
            # model = tlx_densenet121(pretrained=True)
            import densenet
            importlib.reload(densenet)
            model = densenet.densenet121(pretrained=True)
        elif model_name == "rednet50":  # 序号28 - ok TODO - 转换后单独算误差为0
            # from rednet import rednet50 as tlx_rednet50
            # model = tlx_rednet50(pretrained=True)
            import rednet
            importlib.reload(rednet)
            model = rednet.rednet50(pretrained=True)
        elif model_name == "rednet101":  # 序号29 - ok TODO - 转换后单独算误差为0
            # from rednet import rednet101 as tlx_rednet101
            # model = tlx_rednet101(pretrained=True)
            import rednet
            importlib.reload(rednet)
            model = rednet.rednet101(pretrained=True)
        elif model_name == "cspdarknet53":  # 序号2 - TODO - ok
            # from cspdarknet import cspdarknet53 as tlx_cspdarknet53
            # model = tlx_cspdarknet53(pretrained=True)
            import cspdarknet
            importlib.reload(cspdarknet)
            model = cspdarknet.cspdarknet53(pretrained=True)
        elif model_name == "dla34":  # 序号7 - ok
            # from dla import dla34 as tlx_dla34
            # model = tlx_dla34(pretrained=True)
            import dla
            importlib.reload(dla)
            model = dla.dla34(pretrained=True)
        elif model_name == "dla102":  # 序号8 - ok
            # from dla import dla102 as tlx_dla102
            # model = tlx_dla102(pretrained=True)
            import dla
            importlib.reload(dla)
            model = dla.dla102(pretrained=True)
        elif model_name == "dpn68":  # 序号9 - ok
            # from dpn import dpn68 as tlx_dpn68
            # model = tlx_dpn68(pretrained=True)
            import dpn
            importlib.reload(dpn)
            model = dpn.dpn68(pretrained=True)
        elif model_name == "dpn107":  # 序号10 - ok
            # from dpn import dpn107 as tlx_dpn107
            # model = tlx_dpn107(pretrained=True)
            import dpn
            importlib.reload(dpn)
            model = dpn.dpn107(pretrained=True)
        elif model_name == "efficientnet_b1":  # 序号11 -ok
            # from efficientnet import efficientnetb1 as tlx_efficientnetb1
            # model = tlx_efficientnetb1(pretrained=True)
            import efficientnet
            importlib.reload(efficientnet)
            model = efficientnet.efficientnetb1(pretrained=True)
        elif model_name == "efficientnet_b7":  # 序号12 - ok
            # from efficientnet import efficientnetb7 as tlx_efficientnetb7
            # model = tlx_efficientnetb7(pretrained=True)
            import efficientnet
            importlib.reload(efficientnet)
            model = efficientnet.efficientnetb7(pretrained=True)
        elif model_name == "ghostnet":  # 序号13 - ok
            # from ghostnet import ghostnet_x0_5 as tlx_ghostnet_x0_5
            # model = tlx_ghostnet_x0_5(pretrained=True)
            import ghostnet
            importlib.reload(ghostnet)
            model = ghostnet.ghostnet_x0_5(pretrained=True)
        elif model_name == "hardnet39":  # 序号17 - ok
            # from hardnet import hardnet39 as tlx_hardnet39
            # model = tlx_hardnet39(pretrained=True)
            import hardnet
            importlib.reload(hardnet)
            model = hardnet.hardnet39(pretrained=True)
        elif model_name == "hardnet85":  # 序号18 - ok
            # from hardnet import hardnet85 as tlx_hardnet85
            # model = tlx_hardnet85(pretrained=True)
            import hardnet
            importlib.reload(hardnet)
            model = hardnet.hardnet85(pretrained=True)
        elif model_name == "resnest50":  # 序号34 - ok
            # from resnest import resnest50 as tlx_resnest50
            # model = tlx_resnest50(pretrained=True)
            import resnest
            importlib.reload(resnest)
            model = resnest.resnest50(pretrained=True)
        elif model_name == "resnext50":  # 序号35 - ok
            # from resnext import resnext50_32x4d as tlx_resnext50_32x4d
            # model = tlx_resnext50_32x4d(pretrained=True)
            import resnext
            importlib.reload(resnext)
            model = resnext.resnext50_32x4d(pretrained=True)
        elif model_name == "resnext101":  # 序号36 - ok
            # from resnext import resnext101_32x4d as tlx_resnext101_32x4d
            # model = tlx_resnext101_32x4d(pretrained=True)
            import resnext
            importlib.reload(resnext)
            model = resnext.resnext101_32x4d(pretrained=True)
        elif model_name == "rexnet":  # 序号37 - ok
            # from rexnet import rexnet_1_0 as tlx_rexnet_1_0
            # model = tlx_rexnet_1_0(pretrained=True)
            import rexnet
            importlib.reload(rexnet)
            model = rexnet.rexnet_1_0(pretrained=True)  # 0.00061244145
        elif model_name == "se_resnext":  # 序号38 - ok
            # from se_resnext import se_resnext50_32x4d as tlx_se_resnext50_32x4d
            # model = tlx_se_resnext50_32x4d(pretrained=True)
            import se_resnext
            importlib.reload(se_resnext)
            model = se_resnext.se_resnext50_32x4d(pretrained=True)
        elif model_name == "esnet_x0_5":  # 序号47 - ok
            # from esnet import esnet_x0_5 as tlx_esnet_x0_5
            # model = tlx_esnet_x0_5(pretrained=True)
            import esnet
            importlib.reload(esnet)
            model = esnet.esnet_x0_5(pretrained=True)
        elif model_name == "esnet_x1_0":  # 序号48 - ok
            # from esnet import esnet_x1_0 as tlx_esnet_x1_0
            # model = tlx_esnet_x1_0(pretrained=True)
            import esnet
            importlib.reload(esnet)
            model = esnet.esnet_x1_0(pretrained=True)
        elif model_name == "vit":  # 序号20 - ok
            # from vision_transformer import vit_small_patch16_224 as tlx_vit_small_patch16_224
            # model = tlx_vit_small_patch16_224(pretrained=True)
            import vision_transformer
            importlib.reload(vision_transformer)
            model = vision_transformer.vit_small_patch16_224(pretrained=True)
        elif model_name == "alt_gvt_small":  # 序号15 - ok TODO - 转换后单独算误差为0
            # from gvt import alt_gvt_small as tlx_alt_gvt_small
            # model = tlx_alt_gvt_small(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.alt_gvt_small(pretrained=True)
        elif model_name == "alt_gvt_base":  # 序号16 - ok TODO - 转换后单独算误差为0
            # from gvt import alt_gvt_base as tlx_alt_gvt_base
            # model = tlx_alt_gvt_base(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.alt_gvt_base(pretrained=True)
        elif model_name == "pcpvt_base":  # 序号26 - ok TODO - 转换后单独算误差为0
            # from gvt import pcpvt_base as tlx_pcpvt_base
            # model = tlx_pcpvt_base(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.pcpvt_base(pretrained=True)
        elif model_name == "pcpvt_large":  # 序号27 - ok TODO - 转换后单独算误差为0
            # from gvt import pcpvt_large as tlx_pcpvt_large
            # model = tlx_ppcpvt_large(pretrained=True)
            import gvt
            importlib.reload(gvt)
            model = gvt.pcpvt_large(pretrained=True)
        elif model_name == "swin_transformer_base":  # 序号3 - ok TODO - 误差 166.55222 - 需再验证
            # from swin_transformer import swintransformer_base_patch4_window7_224 as tlx_swintransformer_base
            # model = tlx_swintransformer_base(pretrained=True)
            import swin_transformer
            importlib.reload(swin_transformer)
            model = swin_transformer.swintransformer_base_patch4_window7_224(pretrained=True)
        elif model_name == "swin_transformer_small":  # 序号4 - ok TODO - 误差 19.146187
            # from swin_transformer import swintransformer_small_patch4_window7_224 as tlx_swintransformer_small
            # model = tlx_swintransformer_small(pretrained=True)
            import swin_transformer
            importlib.reload(swin_transformer)
            model = swin_transformer.swintransformer_small_patch4_window7_224(pretrained=True)
        elif model_name == "xception41":  # 序号42 - ok
            # from xception import xception41 as tlx_xception41
            # model = tlx_xception41(pretrained=True)
            import xception
            importlib.reload(xception)
            model = xception.xception41(pretrained=True)
        elif model_name == "xception65":  # 序号43 - ok
            # from xception import xception65 as tlx_xception65
            # model = tlx_xception65(pretrained=True)
            import xception
            importlib.reload(xception)
            model = xception.xception65(pretrained=True)
        elif model_name == "xception41_deeplab":  # 序号44 - ok
            # from xception_deeplab import xception41_deeplab as tlx_xception41_deeplab
            # model = tlx_xception41_deeplab(pretrained=True)
            import xception_deeplab
            importlib.reload(xception_deeplab)
            model = xception_deeplab.xception41_deeplab(pretrained=True)
        elif model_name == "xception65_deeplab":  # 序号45 - ok
            # from xception_deeplab import xception65_deeplab as tlx_xception65_deeplab
            # model = tlx_xception65_deeplab(pretrained=True)
            import xception_deeplab
            importlib.reload(xception_deeplab)
            model = xception_deeplab.xception65_deeplab(pretrained=True)
        elif model_name == "levit":  # 序号21 - ok
            # from levit import levit_128 as tlx_levit_128
            # model = tlx_levit_128(pretrained=True)
            import levit
            importlib.reload(levit)
            model = levit.levit_128(pretrained=True)
        elif model_name == "mixnet":  # 序号22 - ok TODO - 误差 0.00048300158
            # from mixnet import mixnet_s as tlx_mixnet_s
            # model = tlx_mixnet_s(pretrained=True)
            import mixnet
            importlib.reload(mixnet)
            model = mixnet.mixnet_s(pretrained=True)
        # todo - new added model
        elif model_name == "convnext":  # 序号103 - TODO
            import convnext
            importlib.reload(convnext)
            model = convnext.convnext(pretrained=True)
        elif model_name == "cswin":  # 序号101 ok
            import cswin_transformer
            importlib.reload(cswin_transformer)
            model = cswin_transformer.CSwintransformer_thiny(pretrained=True)
        elif model_name == "deittiny":  # 序号49 - TODO - 转换后单独算误差为0
            import deit
            importlib.reload(deit)
            model = deit.DeiT_tiny(pretrained=True)
        elif model_name == "deitsmall":  # 序号50 - TODO - 转换后单独算误差为0
            import deit
            importlib.reload(deit)
            model = deit.DeiT_small(pretrained=True)
        elif model_name == "deitbase":  # 序号51 - TODO - 转换后单独算误差为0
            import deit
            importlib.reload(deit)
            model = deit.DeiT_base(pretrained=True)
        elif model_name == "dvt":  # 序号6 TODO -  转换后单独算误差为0
            import distilled_vision_transformer
            importlib.reload(distilled_vision_transformer)
            model = distilled_vision_transformer.Distilled_vision_transformer_base(pretrained=True)
        elif model_name == "peleenet":  # 序号102 ok
            import peleenet
            importlib.reload(peleenet)
            model = peleenet.peleenet(pretrained=True)
        elif model_name == "pp_hgnet":  # 序号106 ok
            import pp_hgnet
            importlib.reload(pp_hgnet)
            model = pp_hgnet.pp_hgnet(pretrained=True)
        elif model_name == "pp_lcnet":  # 序号107 ok
            import pp_lcnet
            importlib.reload(pp_lcnet)
            model = pp_lcnet.pp_lcnet(pretrained=True)
        elif model_name == "pp_lcnet_v2":  # 序号108 ok
            import pp_lcnet_v2
            importlib.reload(pp_lcnet_v2)
            model = pp_lcnet_v2.pp_lcnetv2(pretrained=True)
        elif model_name == "pvt_v2":  # 序号109 ok
            import pvt_v2
            importlib.reload(pvt_v2)
            model = pvt_v2.pvt_v2(pretrained=True)
        elif model_name == "res2net":  # 序号110 ok
            import res2net
            importlib.reload(res2net)
            model = res2net.res2net(pretrained=True)
        elif model_name == "van":  # 序号104 ok
            import van
            importlib.reload(van)
            model = van.van(pretrained=True)
        # sys.path.remove(tlx_project_path)
        return model
