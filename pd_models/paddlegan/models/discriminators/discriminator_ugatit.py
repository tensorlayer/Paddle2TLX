# code was based on https://github.com/znxlwm/UGATIT-pytorch

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class UGATITDiscriminator(nn.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_type='instance'):
        super(UGATITDiscriminator, self).__init__()
        # if norm_type=='instance':
        #     use_bias = True
        # else:
        #     use_bias = False
        model = [
            nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
                nn.Conv2D(input_nc,
                          ndf,
                          kernel_size=4,
                          stride=2,
                          padding=0,
                        #   bias_attr=use_bias
                          ),
            nn.InstanceNorm2D(num_features=ndf),
            nn.LeakyReLU(0.2)
        ]

        for i in range(1, n_layers - 2):
            mult = 2**(i - 1)
            model += [
                nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
                    nn.Conv2D(ndf * mult,
                              ndf * mult * 2,
                              kernel_size=4,
                              stride=2,
                              padding=0,
                            #   bias_attr=use_bias
                              ),
                # norm_layer(ndf * mult * 2),
                nn.InstanceNorm2D(num_features=ndf * mult * 2),
                nn.LeakyReLU(0.2)
            ]

        mult = 2**(n_layers - 2 - 1)
        # print(f"Conv2D.ndf * mult * 2={ndf * mult * 2}")
        model += [
            nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
                nn.Conv2D(ndf * mult,
                          ndf * mult * 2,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                        #   bias_attr=use_bias
                          ),
            nn.InstanceNorm2D(num_features=ndf * mult * 2),
            nn.LeakyReLU(0.2)
        ]

        # Class Activation Map
        mult = 2**(n_layers - 2)
        # print(f"Linear.ndf_mult={ndf * mult}")
        self.gap_fc = nn.Linear(ndf * mult, 1, bias_attr=False)
        # print(f"Linear.ndf_mult={ndf * mult}")
        self.gmp_fc = nn.Linear(ndf * mult, 1, bias_attr=False)
        self.conv1x1 = nn.Conv2D(ndf * mult * 2,
                                 ndf * mult,
                                 kernel_size=1,
                                 stride=1,
                                #  bias_attr=use_bias
                                 )
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.pad = nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect")
        # print(f"Linear.ndf_mult={ndf * mult}")
        self.conv = nn.Conv2D(ndf * mult,
                            1,
                            kernel_size=4,
                            stride=1,
                            padding=0,
                            bias_attr=False)
                            # norm_layer(ndf * mult * 2)])

        self.model = nn.Sequential(*model)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2D(output_size=1)
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2D(output_size=1)

    def forward(self, input):
        x = self.model(input)
        gap = self.adaptive_avg_pool2d(x)
        gap_logit = self.gap_fc(gap.reshape([x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0].transpose([1, 0])
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = self.adaptive_max_pool2d(x)
        gmp_logit = self.gmp_fc(gmp.reshape([x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0].transpose([1, 0])
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = paddle.concat([gap_logit, gmp_logit], 1)
        x = paddle.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = paddle.sum(x, 1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap
