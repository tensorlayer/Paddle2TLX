import os
import wget

# def download_and_decompress(url, path='E:/code/paddle2tlx/auto-paddle2tlx-0105/pretrain/paddlerscd'):
def download_and_decompress(url, path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))):
    if not os.path.exists(path):
        os.mkdir(path)
    fname = os.path.split(url)[-1]
    fullname = os.path.join(path, fname)
    if not os.path.exists(fullname):
        fullname = wget.download(url, out=fullname)
    return fullname