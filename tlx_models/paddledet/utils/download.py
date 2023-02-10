from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import os.path as osp
import shutil
import requests
import hashlib
import base64
import binascii
import sys
import tqdm
import time
from .logger import setup_logger
logger = setup_logger(__name__)
__all__ = ['get_weights_path', 'get_dataset_path', 'download_dataset']
WEIGHTS_HOME = osp.expanduser('~/.cache/paddle/weights/paddledet')
DATASET_HOME = osp.expanduser('~/.cache/paddle/dataset')
CONFIGS_HOME = osp.expanduser('~/.cache/paddle/configs')
DATASETS = {'coco': ([('http://images.cocodataset.org/zips/train2017.zip',
    'cced6f7f71b7629ddf16f17bbcfab6b2'), (
    'http://images.cocodataset.org/zips/val2017.zip',
    '442b8da7639aecaf257c1dceb8ba8c80'), (
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    'f4bbac642086de4f52a3fdda2de5fa2c')], ['annotations', 'train2017',
    'val2017'])}
DOWNLOAD_RETRY_LIMIT = 3
PPDET_WEIGHTS_DOWNLOAD_URL_PREFIX = 'https://paddledet.bj.bcebos.com/'


def parse_url(url):
    url = url.replace('det://', PPDET_WEIGHTS_DOWNLOAD_URL_PREFIX)
    return url


def get_weights_path(url):
    """Get weights path from WEIGHTS_HOME, if not exists,
    download it from url.
    """
    url = parse_url(url)
    path, _ = get_path(url, WEIGHTS_HOME)
    return path


def get_dataset_path(path, annotation, image_dir):
    """
    If path exists, return path.
    Otherwise, get dataset path from DATASET_HOME, if not exists,
    download it.
    """
    if _dataset_exists(path, annotation, image_dir):
        return path
    logger.info(
        'Dataset {} is not valid for reason above, try searching {} or downloading dataset...'
        .format(osp.realpath(path), DATASET_HOME))
    data_name = os.path.split(path.strip().lower())[-1]
    for name, dataset in DATASETS.items():
        if data_name == name:
            logger.debug('Parse dataset_dir {} as dataset {}'.format(path,
                name))
            if name == 'objects365':
                raise NotImplementedError(
                    'Dataset {} is not valid for download automatically. Please apply and download the dataset from https://www.objects365.org/download.html'
                    .format(name))
            data_dir = osp.join(DATASET_HOME, name)
            if name == 'mot':
                if osp.exists(path) or osp.exists(data_dir):
                    return data_dir
                else:
                    raise NotImplementedError(
                        'Dataset {} is not valid for download automatically. Please apply and download the dataset following docs/tutorials/PrepareMOTDataSet.md'
                        .format(name))
            if name == 'spine_coco':
                if _dataset_exists(data_dir, annotation, image_dir):
                    return data_dir
            if name in ['voc', 'fruit', 'roadsign_voc']:
                exists = True
                for sub_dir in dataset[1]:
                    check_dir = osp.join(data_dir, sub_dir)
                    if osp.exists(check_dir):
                        logger.info('Found {}'.format(check_dir))
                    else:
                        exists = False
                if exists:
                    return data_dir
            check_exist = (name != 'voc' and name != 'fruit' and name !=\
                'roadsign_voc')
            for url, md5sum in dataset[0]:
                get_path(url, data_dir, md5sum, check_exist)
            return data_dir
    raise ValueError(
        "Dataset {} is not valid and cannot parse dataset type '{}' for automaticly downloading, which only supports 'voc' , 'coco', 'wider_face', 'fruit', 'roadsign_voc' and 'mot' currently"
        .format(path, osp.split(path)[-1]))


def map_path(url, root_dir, path_depth=1):
    assert path_depth > 0, 'path_depth should be a positive integer'
    dirname = url
    for _ in range(path_depth):
        dirname = osp.dirname(dirname)
    fpath = osp.relpath(url, dirname)
    zip_formats = ['.zip', '.tar', '.gz']
    for zip_format in zip_formats:
        fpath = fpath.replace(zip_format, '')
    return osp.join(root_dir, fpath)


def get_path(url, root_dir, md5sum=None, check_exist=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    url (str): download url
    root_dir (str): root dir for downloading, it should be
                    WEIGHTS_HOME or DATASET_HOME
    md5sum (str): md5 sum of download package
    """
    fullpath = map_path(url, root_dir)
    decompress_name_map = {'VOCtrainval_11-May-2012': 'VOCdevkit/VOC2012',
        'VOCtrainval_06-Nov-2007': 'VOCdevkit/VOC2007',
        'VOCtest_06-Nov-2007': 'VOCdevkit/VOC2007', 'annotations_trainval':
        'annotations'}
    for k, v in decompress_name_map.items():
        if fullpath.find(k) >= 0:
            fullpath = osp.join(osp.split(fullpath)[0], v)
    if osp.exists(fullpath) and check_exist:
        if osp.isfile(fullpath):
            logger.debug('Found {}'.format(fullpath))
            return fullpath, True
        else:
            os.remove(fullpath)
    fullname = _download_dist(url, root_dir, md5sum)
    return fullpath, False


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)
    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0
    while not (osp.exists(fullname) and _check_exist_file_md5(fullname,
        md5sum, url)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError('Download from {} failed. Retry limit reached'
                .format(url))
        logger.info('Downloading {} from {}'.format(fname, url))
        if sys.platform == 'win32':
            url = url.replace('\\', '/')
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError('Downloading from {} failed with code {}!'.
                format(url, req.status_code))
        tmp_fullname = fullname + '_tmp'
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm.tqdm(req.iter_content(chunk_size=1024),
                    total=(int(total_size) + 1023) // 1024, unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
    return fullname


def _download_dist(url, path, md5sum=None):
    env = os.environ
    if 'PADDLE_TRAINERS_NUM' in env and 'PADDLE_TRAINER_ID' in env:
        rank_id_curr_node = int(os.environ.get('PADDLE_RANK_IN_NODE', 0))
        num_trainers = int(env['PADDLE_TRAINERS_NUM'])
        if num_trainers <= 1:
            return _download(url, path, md5sum)
        else:
            fname = osp.split(url)[-1]
            fullname = osp.join(path, fname)
            lock_path = fullname + '.download.lock'
            if not osp.isdir(path):
                os.makedirs(path)
            if not osp.exists(fullname):
                with open(lock_path, 'w'):
                    os.utime(lock_path, None)
                if rank_id_curr_node == 0:
                    _download(url, path, md5sum)
                    os.remove(lock_path)
                else:
                    while os.path.exists(lock_path):
                        time.sleep(0.5)
            return fullname
    else:
        return _download(url, path, md5sum)


def download_dataset(path, dataset=None):
    if dataset not in DATASETS.keys():
        logger.error('Unknown dataset {}, it should be {}'.format(dataset,
            DATASETS.keys()))
        return
    dataset_info = DATASETS[dataset][0]
    logger.debug('Download dataset {} begin.'.format(dataset))
    for info in dataset_info:
        get_path(info[0], path, info[1], False)
    logger.debug('Download dataset {} finished.'.format(dataset))


def _dataset_exists(path, annotation, image_dir):
    """
    Check if user define dataset exists
    """
    if not osp.exists(path):
        logger.warning(
            'Config dataset_dir {} is not exits, dataset config is not valid'
            .format(path))
        return False
    if annotation:
        annotation_path = osp.join(path, annotation)
        if not osp.isfile(annotation_path):
            logger.warning(
                'Config annotation {} is not a file, dataset config is not valid'
                .format(annotation_path))
            return False
    if image_dir:
        image_path = osp.join(path, image_dir)
        if not osp.isdir(image_path):
            logger.warning(
                'Config image_dir {} is not a directory, dataset config is not valid'
                .format(image_path))
            return False
    return True


def _check_exist_file_md5(filename, md5sum, url):
    return _md5check_from_url(filename, url
        ) if md5sum is None and filename.endswith('pdparams') else _md5check(
        filename, md5sum)


def _md5check_from_url(filename, url):
    req = requests.get(url, stream=True)
    content_md5 = req.headers.get('content-md5')
    req.close()
    if not content_md5 or _md5check(filename, binascii.hexlify(base64.
        b64decode(content_md5.strip('"'))).decode()):
        return True
    else:
        return False


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True
    logger.debug('File {} md5 checking...'.format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda : f.read(4096), b''):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()
    if calc_md5sum != md5sum:
        logger.warning('File {} md5 check failed, {}(calc) != {}(base)'.
            format(fullname, calc_md5sum, md5sum))
        return False
    return True


def _move_and_merge_tree(src, dst):
    """
    Move src directory to dst, if dst is already exists,
    merge src to dst
    """
    if not osp.exists(dst):
        shutil.move(src, dst)
    elif osp.isfile(src):
        shutil.move(src, dst)
    else:
        for fp in os.listdir(src):
            src_fp = osp.join(src, fp)
            dst_fp = osp.join(dst, fp)
            if osp.isdir(src_fp):
                if osp.isdir(dst_fp):
                    _move_and_merge_tree(src_fp, dst_fp)
                else:
                    shutil.move(src_fp, dst_fp)
            elif osp.isfile(src_fp) and not osp.isfile(dst_fp):
                shutil.move(src_fp, dst_fp)
