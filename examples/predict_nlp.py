# coding: utf-8
import os
import tensorlayerx as tlx
import numpy as np

word_path = './texts/word_dict.npy'
data_path = './texts/test_text.npy'


def data_process(data, model_name="rnn"):
    def ids_to_str(ids):
        words = []
        for k in ids:
            w = list(word_dict)[k]
            words.append(w if isinstance(w, str) else w.decode('ASCII'))
        return ' '.join(words)

    seq_len = 200
    word_dict = np.load(word_path, allow_pickle=True).item()
    pad_id = word_dict['<pad>']
    padded_sents = []
    contents = []
    for batch_id, d in enumerate(data):
        padded_sent = np.concatenate([d[:seq_len], [pad_id] * (seq_len - len(d))]).astype('int32')
        contents.append(ids_to_str(padded_sent))
        padded_sents.append(tlx.convert_to_tensor(padded_sent))
    return padded_sents, contents


def predict_tlx(model_tlx, model_name="rnn"):
    data = np.load(data_path, allow_pickle=True)
    label_map = {(0): 'negative', (1): 'positive'}
    tests, contents = data_process(data[0], model_name)
    for batch_id, data in enumerate(tests):
        sent = data.unsqueeze(0)
        tlx_results = model_tlx(sent)
        tlx_out = tlx.nn.Softmax()(tlx_results[0]).numpy()
        print(f'TLX_MODEL {model_tlx.__class__.__name__} :')
        print(f'Predicted category:{label_map[np.argmax(tlx_out)]}')
        print(f'negative:{tlx_out[0]},positive:{tlx_out[1]}')
        break
    return tlx_out


def predict_pd(model_pd, model_name="rnn"):
    import paddle
    data = np.load(data_path, allow_pickle=True)
    label_map = {(0): 'negative', (1): 'positive'}
    tests, contents = data_process(data[0], model_name)
    for batch_id, data in enumerate(tests):
        sent = data.unsqueeze(0)
        pd_results = model_pd(sent)
        pd_out = paddle.nn.Softmax()(pd_results[0]).numpy()
        print(f'PD_MODEL {model_pd.__class__.__name__} :')
        print(f'Predicted category:{label_map[np.argmax(pd_out)]}')
        print(f'negative:{pd_out[0]},positive:{pd_out[1]}')
        break
    return pd_out


def calc_diff(result_tlx, result_pd, model_name="rnn"):
    print("=" * 16, " Prediction Error ", "=" * 16)
    print('Model name:', model_name)
    diff = np.abs((np.array(result_tlx) - np.array(result_pd)))
    print('diff sum value:', np.sum(diff))
    return diff
