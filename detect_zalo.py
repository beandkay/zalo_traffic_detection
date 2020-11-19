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
from utils.torch_utils import select_device, load_classifier, time_synchronized
import glob
# from cuong_utils import *
from utils.datasets import *
from predict.post_process import *

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

def detect_coco(model, image, image_meta, opt, device):
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
            boxes = infer(model, img, w_index, h_index, opt, device, half)

            if opt.single_class:
                for bbox in boxes:
                    bbox.cat = 0

            list_image_split[w_index][h_index].bboxes = boxes
    new_img_anno = img_det.merge_boxes(list_image_split, 8, 0.1)
    org, overlaid = new_img_anno.bboxesOnImage(opt.root_dir)
    if opt.save_image:
        cv2.imwrite('{}/{}.png'.format(opt.output_dir, image_meta["image_id"]), overlaid)
    # print(new_img_anno.toJson())
    return new_img_anno.toJson()

def infer(model, img, w_index, h_index, opt,device, half=True):
    # half = device.type != 'cpu'
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

        # print(type(pred), '\n\n')

        det_boxes = []
        for _, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()

                # print(img_.w_index, img_.h_index)
                # print(det)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    box_coord = torch.Tensor(xyxy).flatten().int().tolist()
                    box = toCocoBox(box_coord)
                    # box = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    # box = [int(x) for x in box]
                    box_obj = Bbox(box, int(cls))
                    box_obj.set_part_index(w_index, h_index)
                    box_obj.set_score(float(conf))
                    det_boxes.append(box_obj)
        return det_boxes

def single_image_detect(opt):
    model, device = load_model(opt)
    root_dir = opt.root_dir
    # root_dir  = "/home/asilla/sonnh/yolov5/inference/zalo_test/"
    path = '/home/asilla/sonnh/yolov5/inference/zalo_test/0.png'
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

    for img_name in tqdm(os.listdir(opt.root_dir)):
        # print(img_name)
        # print('dasdsadssa')
        original_image = cv2.imread(os.path.join(opt.root_dir, img_name))
        # print(original_image)
        h,w = original_image.shape[:2]
        img_meta = {"file_name": img_name, "height": h, "width": w, "image_id": int(img_name.split('.')[0])}
        json_ = detect_coco(model, original_image, img_meta, opt, device)
        sample_json += json_

    with open(opt.json_dir, 'w') as outfile:
        json.dump(sample_json, outfile)

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
    root_dir  = "/home/asilla/sonnh/yolov5/inference/zalo_test/"
    path = '/home/asilla/sonnh/yolov5/inference/zalo_test/0.png'
    img_name = "0.png"
    img_path = os.path.join(root_dir, img_name)
    original_image = cv2.imread(img_path)
    h,w = original_image.shape[:2]
    img_meta = {"file_name": img_name, "height": h, "width": w, "image_id": 1}
    list_image_split, no_img_h, no_img_w = split_img(original_image,
                                                    stride = (84, 84),
                                                    save_image = False,
                                                    h = 384, w = 384)
    img_det = ImageDetectionResult(img_meta, root_dir, opt.input_w, opt.input_h, 84, 84)
    for w_index in range(no_img_w):
        for h_index in range(no_img_h):
    # for img_ in list_image_split:
        # path =
            img_ = list_image_split[w_index][h_index]
            if img_ is None:
                continue
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
            det_boxes = []
            for _, det in enumerate(pred):  # detections per image
                im0 = img_.image

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # print(img_.w_index, img_.h_index)
                    # print(det)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        box_coord = torch.Tensor(xyxy).flatten().int().tolist()
                        box = toCocoBox(box_coord)
                        # box = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                        # box = [int(x) for x in box]
                        if (w_index==10 and h_index==2):
                            print(box)
                        box_obj = Bbox(box, int(cls))
                        box_obj.set_part_index(w_index, h_index)
                        det_boxes.append(box_obj)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line) + '\n') % line)

                        if save_img or view_img:  # Add bbox to image
                            # label = '%s %.2f' % (names[int(cls)], conf)
                            label = '%s %.2f' % (int(cls), conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            list_image_split[w_index][h_index].bboxes = det_boxes
            cv2.imwrite('{}/test_{}_{}.jpg'.format(out, img_.w_index, img_.h_index), im0)
            #TODO: ghép ảnh
    new_img_anno = img_det.merge_boxes(list_image_split, 8, 0.2)
    org, overlaid = new_img_anno.bboxesOnImage(root_dir)
    cv2.imwrite('{}/overlaid.jpg'.format(out), overlaid)
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
    parser.add_argument('--output_dir', type=str, default='inference/output_ov', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--stride_w', type=int, default=84, help='stride_w')
    parser.add_argument('--stride_h', type=int, default=84, help='stride_w')
    parser.add_argument('--input_w', type=int, default=384, help='stride_w')
    parser.add_argument('--input_h', type=int, default=384, help='stride_w')
    parser.add_argument('--root_dir', type=str, default="/home/asilla/sonnh/yolov5/inference/zalo_test/", help='directory to save results')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--json_dir', type=str, default="", help='directory to save json  results')
    parser.add_argument('--single_class', action='store_true')
    
    
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