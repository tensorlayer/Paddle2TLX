# coding: utf-8
import os
import paddle
import paddle.nn as nn
import numpy as np
from paddle2tlx.pd2tlx.utils import load_model_nlp

model_urls = {
    'textcnn': os.path.join(os.path.dirname(__file__), '../../pretrain/paddlenlp/textcnn_best_model_final.pdparams')
}


class TextCNN(nn.Layer):
    def __init__(self, vocab_size, embedding_size=256, classes=2, pretrained=None, kernel_num=100, kernel_size=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.pretrained = pretrained
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.dropout = dropout

        if self.pretrained != None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        temp = [(kernel_size_, embedding_size) for kernel_size_ in self.kernel_size]
        self.convs = nn.LayerList([nn.Conv2D(1, self.kernel_num, kernel) for kernel in temp])
        # self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(3 * self.kernel_num, self.classes)

    def forward(self, x):
        embedding = self.embedding(x).unsqueeze(1)
        convs = [nn.ReLU()(conv(embedding)).squeeze(3) for conv in self.convs]
        pool_out = [nn.MaxPool1D(block.shape[2])(block).squeeze(2) for block in convs]
        pool_out = paddle.concat(pool_out, 1)
        logits = self.linear(pool_out)

        return logits


def TextCnn(vocab_size,pretrained,arg, **kwargs):
    model =TextCNN(vocab_size=vocab_size,**kwargs)

    if pretrained:
        model = load_model_nlp(model, arg, model_urls)
    return model


def textcnn(vocab_size, pretrained=False,  **kwargs):

    return TextCnn(vocab_size, pretrained,'textcnn', **kwargs)

