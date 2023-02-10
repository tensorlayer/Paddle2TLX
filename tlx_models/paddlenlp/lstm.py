import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import tensorlayerx.nn as nn
import numpy as np
import tensorlayerx
from paddle2tlx.pd2tlx.utils import restore_model_nlp
model_urls = {
    'lstm': os.path.join(os.path.dirname(__file__), '../../pretrain/paddlenlp/lstm_best_model_final.pdparams')
}


class Lstm(nn.Module):

    def __init__(self, vocab_size, embedding_size=256):
        super(Lstm, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(256, 256, bias=True, batch_first=True,
            bidirectional=False)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=2,
            bias=True, batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=2)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        output, (hidden, _) = self.lstm(emb)
        x = tensorlayerx.reduce_mean(output, axis=1)
        return self.linear(x)


def lstm_pd(vocab_size, pretrained, arg, **kwargs):
    model = Lstm(vocab_size=vocab_size, **kwargs)
    if pretrained:
        model = restore_model_nlp(model, arg, model_urls)
    return model


def lstm(vocab_size, pretrained=False, **kwargs):
    return lstm_pd(vocab_size, pretrained, 'lstm', **kwargs)
