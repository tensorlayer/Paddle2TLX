import tensorlayerx as tlx
import paddle
import paddle2tlx
import sys
import os.path as osp
import logging
parent_path = osp.abspath(osp.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)
from utils.download import download_dataset
logging.basicConfig(level=logging.INFO)
download_path = osp.split(osp.realpath(sys.argv[0]))[0]
download_dataset(download_path, 'coco')
