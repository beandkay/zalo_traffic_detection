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

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import glob
from cuong_utils import *
from utils.datasets import *

def split_img(original_image,
                stride = (84, 84), 
                save_image = False,
                h = 384, w = 384):
    
    list_split_image = []
    for i in range(math.ceil((1622-w)/stride[0]) + 1):
        for j in range(math.ceil((626-h)/stride[1]) + 1):

            if i*stride[0] + w < 1622:
                xmin = i*stride[0]
                xmax = i*stride[0] + w
            else:
                xmin = 1622 - w
                xmax = 1622
            if j*stride[1]  + h< 626:
                ymin = j*stride[1]
                ymax = j*stride[1] + h
            else:
                ymin = 626 - 384
                ymax = 626

            split_bbox = [xmin, ymin, xmax, ymax]

            split_image = copy.deepcopy(original_image[split_bbox[1]:split_bbox[3], split_bbox[0]:split_bbox[2]])
            list_split_image.append(PartImageDetectionResult(split_image, i, j, []))
    return list_split_image


# def load_img(path):
#     img = cv2.imread(path)

def preprocess(img, img_size):
    img = letterbox(img, new_shape=img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    # out = 'inference/output'
    # source = 'inference/images'
    # weights = 'yolov5x.pt'
    # view_img = False
    # save_txt = True
    # imgsz = 384

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    save_img = True

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # for path, list_image_split, im0s in dataset:

    path = '/home/asilla/sonnh/yolov5/inference/zalo_test/0.png'
    original_image = cv2.imread(path)
    list_image_split = split_img(original_image,
                                stride = (84, 84), 
                                save_image = False,
                                h = 384, w = 384)

    for img_ in list_image_split:
        # path = 
        img = img_.image
        img = preprocess(img, imgsz)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = img_.image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                print(img_.w_index, img_.h_index)
                print(det)
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    print(xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s %.2f' % (int(cls), conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
        cv2.imwrite('{}/test_{}_{}.jpg'.format(out, img_.w_index, img_.h_index), im0)
        #TODO: ghép ảnh

    print('Done. (%.3fs)' % (time.time() - t0))
    
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
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()