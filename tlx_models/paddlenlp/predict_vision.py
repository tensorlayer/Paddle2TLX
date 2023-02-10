import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx
import numpy as np


class IMDBDataset(tensorlayerx.dataflow.Dataset):

    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.sents)


def data_process(data):

    def ids_to_str(ids):
        words = []
        for k in ids:
            w = list(word_dict)[k]
            words.append(w if isinstance(w, str) else w.decode('ASCII'))
        return ' '.join(words)
    seq_len = 200
    word_dict = np.load('text/word_dict.npy', allow_pickle=True).item()
    pad_id = word_dict['<pad>']
    padded_sents = []
    contents = []
    for batch_id, d in enumerate(data):
        padded_sent = np.concatenate([d[:seq_len], [pad_id] * (seq_len -
            len(d))]).astype('int32')
        contents.append(ids_to_str(padded_sent))
        padded_sents.append(tensorlayerx.convert_to_tensor(padded_sent))
    return padded_sents, contents


def pd_tlx_diff(pd_model, tlx_model, data):
    label_map = {(0): 'negative', (1): 'positive'}
    pd_results = pd_f(pd_model, data)
    tlx_results = tlx_f(tlx_model, data)
    print(f'Model {pd_model.__class__.__name__} predict category - TLX:',
        label_map[np.argmax(pd_results)])
    print(f'Model {tlx_model.__class__.__name__} predict category - Paddle:',
        label_map[np.argmax(tlx_results)])
    diff = np.fabs(np.array(pd_results) - np.array(tlx_results))
    print('diff sum value:', np.sum(diff))
    print('diff max value:', np.max(diff))


def pd_f(model_pd, data):
    label_map = {(0): 'negative', (1): 'positive'}
    tests, contents = data_process(data)
    for batch_id, data in enumerate(tests):
        sent = data.unsqueeze(0)
        pd_results = model_pd(sent)
        pd_out = tensorlayerx.nn.Softmax()(pd_results[0]).numpy()
        print(f'PD_MODEL {model_pd.__class__.__name__} :')
        print(f'Predicted category:{label_map[np.argmax(pd_out)]}')
        print(f'negative:{pd_out[0]},positive:{pd_out[1]}')
        break
    return pd_out


def tlx_f(model_tlx, data):
    label_map = {(0): 'negative', (1): 'positive'}
    tests, contents = data_process(data)
    for batch_id, data in enumerate(tests):
        sent = data.unsqueeze(0)
        tlx_results = model_tlx(sent)
        tlx_out = tensorlayerx.nn.Softmax()(tlx_results[0]).numpy()
        print(f'TLX_MODEL {model_tlx.__class__.__name__} :')
        print(f'Predicted category:{label_map[np.argmax(tlx_out)]}')
        print(f'negative:{tlx_out[0]},positive:{tlx_out[1]}')
        break
    return tlx_out
