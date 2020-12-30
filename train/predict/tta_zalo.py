import argparse
import os
import shutil
import time
from pathlib import Path
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from TTA import TTAHorizontalFlip, TTAVerticalFlip, TTARotate90, TTACompose
from utils.torch_utils import select_device, load_classifier, time_synchronized
import glob
# from cuong_utils import *
from utils.datasets import *
from post_process import Bbox, PartImageDetectionResult, ImageDetectionResult
from post_process import *
from itertools import product
from ensemble_boxes import *


def split_img(original_image,
                stride = (84, 84), 
                save_image = False,
                h = 384, w = 384):
    img_h, img_w, _ = original_image.shape
    no_img_h = math.ceil((img_h-h)/stride[1]) + 1
    no_img_w = math.ceil((img_w-w)/stride[0]) + 1
    # list_split_image = []
    list_split_image = [[None for _ in range(no_img_h)] for _ in range(no_img_w)]
    for i in range(no_img_w):
        for j in range(no_img_h):

            if i*stride[0] + w < img_w:
                xmin = i*stride[0]
                xmax = i*stride[0] + w
            else:
                xmin = img_w - w
                xmax = img_w
            if j*stride[1]  + h< img_h:
                ymin = j*stride[1]
                ymax = j*stride[1] + h
            else:
                ymin = img_h - h
                ymax = img_h

            split_bbox = [xmin, ymin, xmax, ymax]
            split_image = copy.deepcopy(original_image[split_bbox[1]:split_bbox[3], split_bbox[0]:split_bbox[2]])
            list_split_image[i][j] = PartImageDetectionResult(split_image, i, j, [])
    return list_split_image, no_img_h, no_img_w


# def load_img(path):
#     img = cv2.imread(path)

def preprocess(img, img_size):
    img = letterbox(img, new_shape=img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img

def load_model(opt):
    set_logging()
    device = select_device(opt.device)
    if not os.path.exists(opt.output_dir):  # output dir
        os.makedirs(opt.output_dir)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    return model.eval(), device

def detect_coco(model, image, image_meta, opt, device, tta_transforms=None, save_image = True):
    # device = select_device(opt.device)
    half = device.type != 'cpu'
    names = model.module.names if hasattr(model, 'module') else model.names
    list_image_split, no_img_h, no_img_w = split_img(image,
                                stride = (opt.stride_w, opt.stride_h),
                                save_image = False,
                                h = opt.input_h, w = opt.input_w)
    img_det = ImageDetectionResult(image_meta, opt.root_dir, opt.input_w, opt.input_h, opt.stride_w, opt.stride_h)
    for w_index in range(no_img_w):
        for h_index in range(no_img_h):
            img_obj = list_image_split[w_index][h_index]
            if img_obj is None:
                continue
            img = img_obj.image
            if tta_transforms is None:
                boxes = infer(model, img, w_index, h_index, opt, device, half)
            else:
                boxes = infer_tta(model, img, w_index, h_index, opt, device, tta_transforms)
            list_image_split[w_index][h_index].bboxes = boxes
    new_img_anno = img_det.merge_boxes_v2(list_image_split, 8, 0.1)
    org, overlaid = new_img_anno.bboxesOnImage(opt.root_dir)
    if save_image:
        cv2.imwrite('{}/{}.png'.format(opt.output_dir, image_meta["image_id"]), overlaid)
    # print(new_img_anno.toJson())
    return new_img_anno.toJson()

def infer(model, img, w_index, h_index, opt,device, tta_transforms=None):
    half = device.type != 'cpu'
    with torch.no_grad():
        original_image = img.copy()
        img = preprocess(img, opt.img_size)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # print(pred)
        det_boxes = []
        for _, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    box_coord = torch.Tensor(xyxy).flatten().int().tolist()
                    box = toCocoBox(box_coord)
                    box_obj = Bbox(box, int(cls))
                    box_obj.set_part_index(w_index, h_index)
                    box_obj.set_score(float(conf))
                    det_boxes.append(box_obj)
        return det_boxes

def infer_img(model, img_tensor, opt, original_image, tta_transform):
    pred = model(img_tensor, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    result = []

    for _, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], original_image.shape).round()
            # print(det[:, :4])
            det = det.cpu().numpy()
            det[:, :4] = tta_transform.deaugment_boxes(det[:, :4].copy())
            # print(det[:, :4])
            scores = det[:, 4]
            class_id = det[:, -1]
            result.append({
                    'boxes': det[:, :4],
                    'scores': scores,
                    'class_id' : class_id
                })
    return result

def run_wbf(predictions, image_size=512, iou_thr=0.2, skip_box_thr=0.4, weights=None):
    boxes = [prediction[0]['boxes']/(image_size) for prediction in predictions]
    scores = [prediction[0]['scores'] for prediction in predictions]
    labels = [prediction[0]['class_id'] for prediction in predictions]
    # print(np.array(boxes).shape)
    # print(np.array(scores).shape)
    # print(np.array(labels).shape)
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    return boxes*image_size, scores, labels
def infer_tta(model, img, w_index, h_index, opt, device, tta_transforms=None):
    half = device.type != 'cpu'
    with torch.no_grad():
        original_image = img.copy()
        img = preprocess(img, opt.img_size)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        dets = []
        if tta_transforms is not None:
            for tta_transform in tta_transforms[:4]:
                img_tensor = tta_transform.batch_augment(img.clone())
                result = infer_img(model, img_tensor, opt, original_image, tta_transform)
                if len(result) > 0:
                    dets.append(result)
        boxes, scores, labels = run_wbf(dets, image_size=opt.img_size, iou_thr=0.55, skip_box_thr=0.7, weights=None)
        boxes = torch.from_numpy(boxes)
        # print(pred)
        det_boxes = []
        for i in range(boxes.shape[0]):
            box_coord = boxes[i].numpy().flatten().astype(np.int).tolist()
            box = toCocoBox(box_coord)
            box_obj = Bbox(box, int(labels[i]))
            box_obj.set_part_index(w_index, h_index)
            box_obj.set_score(float(scores[i]))
            det_boxes.append(box_obj)
        return det_boxes


def single_image_detect(opt):
    model, device = load_model(opt)
    root_dir = opt.root_dir
    # root_dir  = "sonnh/yolov5/inference/zalo_test/"
    path = 'sonnh/yolov5/inference/zalo_test/0.png'
    img_name = "0.png"
    img_path = os.path.join(root_dir, img_name)
    original_image = cv2.imread(img_path)
    h,w = original_image.shape[:2]
    img_meta = {"file_name": img_name, "height": h, "width": w, "image_id": 1}
    detect_coco(model, original_image, img_meta, opt, device)

def zalo_image_detect(opt):
    model, device = load_model(opt)
    sample_json = []
    # {
    #     "image_id": 0,
    #     "category_id": 2,
    #     "bbox": [
    #         1135.73,
    #         283.99,
    #         58.88,
    #         55.51
    #     ],
    #     "score": 0.8389999866485596
    # },
    tta_transforms = []
    print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk", opt.use_tta)
    for tta_combination in product([TTAHorizontalFlip(opt.img_size), None],
                                [TTARotate90(opt.img_size), None]):
        tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
    # print(tta_transforms)
    if not opt.use_tta:
        tta_transforms = None
    for img_name in tqdm(os.listdir(opt.root_dir)):
        # print(img_name)
        # print('dasdsadssa')
        original_image = cv2.imread(os.path.join(opt.root_dir, img_name))
        # print(original_image)
        h,w = original_image.shape[:2]
        img_meta = {"file_name": img_name, "height": h, "width": w, "image_id": int(img_name.split('.')[0])}
        json_ = detect_coco(model, original_image, img_meta, opt, device, tta_transforms)
        sample_json += json_

    with open('first.json', 'w') as outfile:
        json.dump(sample_json, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--output_dir', type=str, default='inference/output_ov', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--stride_w', type=int, default=84, help='stride_w')
    parser.add_argument('--stride_h', type=int, default=84, help='stride_w')
    parser.add_argument('--input_w', type=int, default=384, help='stride_w')
    parser.add_argument('--input_h', type=int, default=384, help='stride_w')
    parser.add_argument('--use_tta', action='store_true', default=False, help='use_tta')
    parser.add_argument('--root_dir', type=str, default="sonnh/yolov5/inference/zalo_test/", help='directory to save results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                # detect()
                single_image_detect(opt)
                strip_optimizer(opt.weights)
        else:
            # detect()
            # single_image_detect(opt)
            zalo_image_detect(opt)