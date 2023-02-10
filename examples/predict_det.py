# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
import sys
import numpy as np
import glob

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
# from paddle2tlx.common.convert import load_inference_model
# from validation import load_inference_model_tlx

def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, '--infer_img or --infer_dir should be set'
    assert infer_img is None or os.path.isfile(infer_img), '{} is not a file'.format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), '{} is not a directory'.format(infer_dir)
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]
    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), 'infer_dir {} is not a directory'.format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, 'no image found in {}'.format(infer_dir)
    return images


def predict_pd(trainer, model_name="YOLOV3", project_src_path=None):
    pd_project_path = project_src_path
    sys.path.insert(0, pd_project_path)
    os.chdir(pd_project_path)
    import configparser
    from tools.infer_det import parse_args

    print("=" * 16, "Predict value in forward propagation - PaddlePaddle", "=" * 16)
    print('Model name:', model_name)
    config = configparser.ConfigParser()
    config.read('/home/sthq/scc/paddle2tlx/my_project/translated_models.cfg')
    config_file = config.get('MODEL_CONFIG_PATH', model_name)
    weights_file = config.get('MODEL_WEIGHTS_PATH', model_name)
    FLAGS = parse_args(config_file, weights_file)

    trainer.load_weights(FLAGS.pretrain_weights)
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    results = trainer.predict(images,
                    output_dir=FLAGS.output_dir,)
    res = 'bbox' if model_name != 'SOLOv2' else 'segm'
    results = np.array(results[0][res])
    print('Predicted value:', results)
    return results


def predict_tlx(trainer, model_name="YOLOV3", project_dst_path=None):
    # tlx_project_path = f"/home/sthq/scc/paddle2tlx/my_project/paddleclas/{model_name}"
    tlx_project_path = project_dst_path
    sys.path.insert(0, tlx_project_path)
    os.chdir(tlx_project_path)
    import configparser
    from tools.infer_det import parse_args

    print("=" * 16, "Predict value in forward propagation - TensorLayerX", "=" * 16)
    print('Model name:', model_name)
    config = configparser.ConfigParser()
    config.read('/home/sthq/scc/paddle2tlx/my_project/translated_models.cfg')
    config_file = config.get('MODEL_CONFIG_PATH', model_name)
    weights_file = config.get('MODEL_WEIGHTS_PATH', model_name)
    FLAGS = parse_args(config_file, weights_file)

    trainer.load_weights(FLAGS.pretrain_weights)
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    results = trainer.predict(images,
                    output_dir=FLAGS.output_dir,)
    res = 'bbox' if model_name != 'SOLOv2' else 'segm'
    results = np.array(results[0][res])
    print('Predicted value:', results)
    return results


def calc_diff(result_tlx, result_pd, model_name="yolov3"):
    print("=" * 16, " Prediction Error ", "=" * 16)
    print('Model name:', model_name)
    diff = np.abs((np.array(result_tlx) - np.array(result_pd)))
    print('diff sum value:', np.sum(diff))
    return diff


if __name__ == '__main__':
    project_src_path = '/home/sthq/scc/paddle2tlx/my_project/paddlers/paddledetection'
    project_dst_path = '/home/sthq/scc/paddle2tlx/my_project/paddleclas/paddledetection'
    model_name = 'YOLOX'
    # tlx_model, pd_model = load_inference_model(project_src_path, project_dst_path, model_name, 'det')
    # result_tlx = predict_tlx(tlx_model, model_name, project_dst_path)
    # result_pd = predict_pd(pd_model, model_name, project_src_path)
    # calc_diff(result_tlx, result_pd, model_name)
