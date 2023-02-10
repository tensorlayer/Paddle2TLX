import os
import paddle.nn as nn
import numpy as np
import paddle
from paddle2tlx.pd2tlx.utils import load_model_nlp

model_urls = {
    'rnn': os.path.join(os.path.dirname(__file__), '../../pretrain/paddlenlp/rnn_best_model_final.pdparams')
}


class Rnn(nn.Layer):
    def __init__(self, vocab_size, embedding_size=256):
        super(Rnn, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.rnn = nn.SimpleRNN(256, 256)
        self.attention = nn.MultiHeadAttention(embed_dim=256, num_heads=2,)  # embed_dim要能被num_heads整除
        self.linear = nn.Linear(in_features=256, out_features=2)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        output, hidden = self.rnn(emb)
        x = paddle.mean(output, axis=1)

        return self.linear(x)


def RNN(vocab_size,pretrained,arg, **kwargs):
    model = Rnn(vocab_size=vocab_size, **kwargs)

    if pretrained:
        model = load_model_nlp(model, arg, model_urls)
    return model


def rnn(vocab_size, pretrained=False,  **kwargs):

    return RNN(vocab_size, pretrained, 'rnn', **kwargs)