import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import tensorlayerx
import tensorlayerx.nn as nn
import numpy as np
from paddle2tlx.pd2tlx.utils import restore_model_nlp
model_urls = {
    'textcnn': os.path.join(os.path.dirname(__file__), '../../pretrain/paddlenlp/textcnn_best_model_final.pdparams')
}


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_size=256, classes=2,
        pretrained=None, kernel_num=100, kernel_size=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.pretrained = pretrained
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.dropout = dropout
        if self.pretrained != None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                embedding_dim=self.embedding_size)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                embedding_dim=self.embedding_size)
        temp = [(kernel_size_, embedding_size) for kernel_size_ in self.
            kernel_size]
        self.convs = nn.ModuleList([nn.GroupConv2d(in_channels=1,
            out_channels=self.kernel_num, kernel_size=kernel, padding=0,
            data_format='channels_first') for kernel in temp])
        self.linear = nn.Linear(in_features=3 * self.kernel_num,
            out_features=self.classes)

    def forward(self, x):
        embedding = self.embedding(x).unsqueeze(1)
        convs = [nn.ReLU()(conv(embedding)).squeeze(3) for conv in self.convs]
        pool_out = [nn.MaxPool1d(block.shape[2], padding=0, data_format=\
            'channels_first')(block).squeeze(2) for block in convs]
        pool_out = tensorlayerx.concat(pool_out, 1)
        logits = self.linear(pool_out)
        return logits


def TextCnn(vocab_size, pretrained, arg, **kwargs):
    model = TextCNN(vocab_size=vocab_size, **kwargs)
    if pretrained:
        model = restore_model_nlp(model, arg, model_urls)
    return model


def textcnn(vocab_size, pretrained=False, **kwargs):
    return TextCnn(vocab_size, pretrained, 'textcnn', **kwargs)
