# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import unittest
from examples.predict_vision import predict_tlx
from examples.predict_vision import calc_diff


image_file = "examples/images/dog.jpeg"
tlx_project_path = "D:/ProjectByPython/code/myproject/model-convert-tools/convert_test/tlx_models"
import sys
sys.path.insert(0, tlx_project_path)


class ConvertProjectInferenceTest(unittest.TestCase):
    # pass
    def test_vgg(self):
        from vgg import vgg16
        model = vgg16(pretrained=True)
        predict_tlx(model, image_file, "vgg16")

    # pass
    def test_alexnet(self):
        from alexnet import alexnet
        model = alexnet(pretrained=True)
        predict_tlx(model, image_file, "alexnet")

    #
    def test_resnet(self):
        from resnet import resnet50
        model = resnet50(pretrained=True)
        predict_tlx(model, image_file, "resnet")


if __name__ == '__main__':
    unittest.main()
