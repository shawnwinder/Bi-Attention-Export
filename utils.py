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



##############################################################################
# model utils
##############################################################################
def load_model(model, init_net_pb, predict_net_pb):
    ''' load init and predict net from .pb file for model validation/testing
        model: current model
        init_net: the .pb file of the init_net
        predict_net: the .pb file of the predict_net
    '''
    # Make sure both nets exists
    if (not os.path.exists(init_net_pb)) or (not os.path.exists(predict_net_pb)):
            print("ERROR: input net.pb not found!")

    # Append net
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'r') as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = model.param_init_net.AppendNet(core.Net(init_net_proto))

    predict_net_proto = caffe2_pb2.NetDef()
    with open(predict_net_pb, 'r') as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = model.net.AppendNet(core.Net(predict_net_proto))


def load_init_net(init_net_pb, device_opt):
    ''' load params of pretrained init_net on given device '''
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'rb') as f:
        init_net_proto.ParseFromString(f.read())
        for op in init_net_proto.op:
            op.device_option.CopyFrom(device_opt)
    workspace.RunNetOnce(core.Net(init_net_proto))


##############################################################################
# image utils
##############################################################################
def rescale(image, SCALE):
    H, W, = image.shape[:-1]
    aspect = float(W) / H
    if aspect > 1:
        w = int(aspect * SCALE)
        img_scaled = cv2.resize(image, (w, SCALE))  # size=(W, H) in opencv
    elif aspect < 1:
        h = int(SCALE / aspect)
        img_scaled = cv2.resize(image, (SCALE, h))
    else:
        img_scaled = cv2.resize(image, (SCALE, SCALE))
    return img_scaled

def central_crop(image, CROP):
    h, w, c = image.shape
    assert(c == 3)
    h_beg = (h - CROP) // 2
    w_beg = (w - CROP) // 2
    return image[h_beg:(h_beg+CROP), w_beg:(w_beg+CROP), :]

def normalize_channel(image, MEAN, STD):
    return (image - MEAN) / STD

def color_jitter(img, JITTER):
    if JITTER:
        # img = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB)  # cv2 defaul color code is BGR
        h, w, c = img.shape
        noise = np.random.randint(0, 50, (h, w)) # design jitter/noise here
        zitter = np.zeros_like(img)
        zitter[:,:,1] = noise
        noise_added = cv2.add(img, zitter)

        combined = np.vstack((img[:h,:,:], noise_added[h:,:,:]))
        return combined
    else:
        return img


if __name__ == "__main__":
    pass



