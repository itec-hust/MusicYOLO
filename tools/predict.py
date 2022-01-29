#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import json
import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, vis

from util import cut_image, get_res

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("MusicYOLO")
    parser.add_argument("--audiodir", type=str)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--ext", type=str, default='.flac')
    parser.add_argument("--prefix", type=bool, default=False)

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.6, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def predict(self, output, img_info, save_dir, cls_conf=0.35):
        ratio = img_info["ratio"]
        file_name = img_info["file_name"]
        height, width = img_info["height"], img_info["width"]
        ext = os.path.splitext(file_name)[1]

        if output is None:
            jsonname = file_name.replace(ext, '.json')
            jsonpath = os.path.join(save_dir, jsonname)
            data = {"file_name": file_name,
                    "img_size": [height, width],
                    "boxs": []}
            with open(jsonpath, 'wt') as f:
                f.write(json.dumps(data))
            return

        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        bboxs = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            score = scores[i]
            if score < cls_conf:
                continue
            bboxs.append([float(co) for co in box])

        jsonname = file_name.replace(ext, '.json')
        jsonpath = os.path.join(save_dir, jsonname)
        data = {"file_name": file_name,
                "img_size": [height, width],
                "boxs": bboxs}
        with open(jsonpath, 'wt') as f:
            f.write(json.dumps(data))


def process_image(predictor, args):
    files = get_image_list(args.imagedir)
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        predictor.predict(outputs[0], img_info, args.jsondir, predictor.confthre)


def main(exp, args):

    image_dir = os.path.join(args.savedir, 'images')
    json_dir = os.path.join(args.savedir, 'json')
    res_dir = os.path.join(args.savedir, 'res')

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    args.imagedir = image_dir
    args.jsondir = json_dir
    args.resdir = res_dir

    logger.info("Args: {}".format(args))

    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    exp.test_size = (args.tsize, args.tsize)

    # generate images
    # cut_image.generate_slices(args.audiodir, image_dir, args.prefix, args.ext)

    # load model
    # model = exp.get_model()
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # if args.device == "gpu":
    #     model.cuda()
    # model.eval()

    # ckpt_file = args.ckpt
    # logger.info("loading checkpoint")
    # ckpt = torch.load(ckpt_file, map_location="cpu")
    # model.load_state_dict(ckpt["model"])
    # logger.info("loaded checkpoint done.")

    # # generate bounding box
    # predictor = Predictor(model, exp, COCO_CLASSES, args.device)
    # current_time = time.localtime()
    # process_image(predictor, args)

    # generate transcriptin results
    get_res.generate_res(args.audiodir, args.resdir, args.ext, args.jsondir, args.prefix)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)

    main(exp, args)