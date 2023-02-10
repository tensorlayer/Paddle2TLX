# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import importlib


class TLXClassificationModel(object):
    def __init__(self, project_path, model_name="rnn"):
        self.model_name = model_name
        self.tlx_model = self.load_tlx_model(project_path, model_name)

    def load_tlx_model(self, tlx_project_path, model_name="rnn"):
        import os
        os.environ["TL_BACKEND"] = "paddle"
        sys.path.insert(0, tlx_project_path)
        model = None
        vocab_size = 5149
        if model_name == "rnn":
            # from vgg import vgg16 as pd_vgg16
            # model = pd_vgg16(pretrained=True)
            import rnn
            importlib.reload(rnn)
            model = rnn.rnn(vocab_size=vocab_size, pretrained=True)
        elif model_name == "lstm":
            import lstm
            importlib.reload(lstm)
            model = lstm.lstm(vocab_size=vocab_size, pretrained=True)
        elif model_name == "textcnn":
            import textcnn
            importlib.reload(textcnn)
            model = textcnn.textcnn(vocab_size=vocab_size, pretrained=True)
        return model

