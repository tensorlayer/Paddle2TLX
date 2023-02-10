import os
import paddle.nn as nn
import numpy as np
import paddle
from paddle2tlx.pd2tlx.utils import load_model_nlp

model_urls = {
    'lstm': os.path.join(os.path.dirname(__file__), '../../pretrain/paddlenlp/lstm_best_model_final.pdparams')
}


class Lstm(nn.Layer):

    def __init__(self, vocab_size, embedding_size=256):
        super(Lstm, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(256, 256)
        self.attention = nn.MultiHeadAttention(embed_dim=256, num_heads=2,)  # embed_dim要能被num_heads整除
        self.linear = nn.Linear(in_features=256, out_features=2)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        output, (hidden, _) = self.lstm(emb)
        x = paddle.mean(output, axis=1)
        return self.linear(x)


def lstm_pd(vocab_size, pretrained, arg, **kwargs):
    model = Lstm(vocab_size=vocab_size, **kwargs)
    if pretrained:
        model = load_model_nlp(model, arg, model_urls)
    return model


def lstm(vocab_size, pretrained=False,  **kwargs):

    return lstm_pd(vocab_size, pretrained, 'lstm', **kwargs)