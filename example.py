from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import argparse
import yaml
import os
import sys
import cv2
import time
import random

from caffe2.python import (
    workspace,
    model_helper,
    core, brew,
    optimizer,
    net_drawer
)
from caffe2.proto import caffe2_pb2

from utils import *


def parse_args():
    # load config file
    config_parser = argparse.ArgumentParser(
        description='Bi-Attention image prediction config parser',
    )
    config_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help = 'config file'
    )
    args = config_parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
        config['config_path'] = os.path.join(os.getcwd(), args.config)
    return config


def add_input(config):
    ''' preprocessing single or batch input images for class prediction
        we store every image path as a line in a file
        we can add or delete image by adding or deleting the image file lines
    Args:
        config dict
    Return:
        np.ndarray images
    '''
    # image input
    image_batch = []
    with open(config['image_files'], 'r') as f:
        for img_path in f:
            # read image
            img_path = img_path.strip()
            assert(os.path.exists(img_path))
            image = cv2.imread(img_path)

            # preprocessing pipeline
            image_norm = normalize_channel(image, config['mean'], config['std'])
            image_rescaled = rescale(image_norm, config['scale'])
            image_cropped = central_crop(image_rescaled, config['crop'])
            image_jittered = color_jitter(image_cropped, config['jitter'])

            # caffe2 formatting
            img = image_jittered.transpose(2, 0, 1) # HWC->CHW

            image_batch.append(img)
    return np.array(image_batch).astype(np.float32)


def add_predictor(config, images):
    ''' predict the label of input single or batch images
    Args:
        config dict
        np.arrary images
    Returns:
        batch softmax results; type: np.array; shape: [prediction batchsize,
        prediction classes]
    '''
    # set device
    device_opt = caffe2_pb2.DeviceOption()
    if config['gpu_id'] is not None:
        device_opt.device_type = caffe2_pb2.CUDA
        device_opt.cuda_gpu_id = config['gpu_id']

    # add prediction model
    predict_model = model_helper.ModelHelper(
        name="predictor",
        init_params=False,
    )

    # load param_init_net
    init_net_proto = caffe2_pb2.NetDef()
    with open(config['init_net'], 'rb') as f:
        init_net_proto.ParseFromString(f.read())
        for op in init_net_proto.op:
            op.device_option.CopyFrom(device_opt)
        workspace.RunNetOnce(core.Net(init_net_proto))

    # load predict_net
    predict_net_proto = caffe2_pb2.NetDef()
    with open(config['predict_net'], 'rb') as f:
        predict_net_proto.ParseFromString(f.read())
        for op in predict_net_proto.op:
            op.device_option.CopyFrom(device_opt)
        predict_model.net = core.Net(predict_net_proto)

    # feed pre-processed images as input
    workspace.FeedBlob("data", images, device_option=device_opt)

    # run net
    workspace.CreateNet(predict_model.net)
    workspace.RunNet(predict_model.net)
    results = workspace.FetchBlob("softmax")

    return results


def get_label_words_mapping(words_file):
    mapping = {}
    with open(words_file, 'r') as f:
        for line in f:
            label, word = line.split(" ", 1)
            mapping[int(label)] = word.strip()
    return mapping


def explain_results(config, results):
    ''' make the prediction results more readalbe to human beings
    '''
    mapping = get_label_words_mapping(config['label_words'])
    images = []
    with open(config['image_files'], 'r') as f:
        images = [line.strip() for line in f]

    print("The image prediction results is as follow:")
    for i in range(results.shape[0]):
        print("="*100)
        softmax = results[i]
        label = np.argmax(softmax)
        word = mapping[label]
        prob = softmax[label]
        print("Image: {}\nLabel: {}\nClass: {}\nConfidence: {:.2f}%".format(
            images[i], label, word, prob*100))
    print("="*100)


def predict_images(config):
    ''' image predictin pipeline
    '''
    images = add_input(config)
    results = add_predictor(config, images)
    explain_results(config, results)


if __name__ == "__main__":
    config = parse_args()
    predict_images(config)






