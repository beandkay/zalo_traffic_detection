import torch
import numpy as np
import json
import os
import cv2

class Bbox:
    def __init__(self, box, cat):
        '''
        @param box [x, y, w, h]
        @param cat class id
        '''
        self.box = box
        self.cat = cat

    def toCocoBoxFormat(self):
        return bbox

    def toConnerForm(self):
        '''
        convert [x, y, w, h] -> [x1, y1, x2, y2]
        '''
        return cocoToAbsoluteBox(self.box)
    
    def onImage(self, image):
        '''
        draw bbox on image
        '''
        start_point = (self.box[0], self.box[1])
        end_point = (self.box[0]+self.box[2], self.box[1]+self.box[3])
        overlaid_img = cv2.rectangle(image.copy(), start_point , end_point, COLOR_CLASS[self.cat], 2)
        return overlaid_img

class PartImageDetectionResult:
    '''
    @param image cropped image
    @param w_index index of image along width (start from 0)
    @param h_index index of image along height (start from 0)
    @bboxes Bbox object
    '''
    def __init__(self, image, w_index, h_index, bboxes):
        self.image = image
        self.w_index = w_index
        self.h_index = h_index
        self.bboxes = bboxes

    def drawBboxOnImage(self):
        img = self.image.copy()
        for box in self.bboxes:
            img = box.onImage(img)
        return img

def getOverlapBox(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return []
    else:
        return [x1, y1, width, height]
def transformCoordinate(coordinate, stride, idx):
    return (coordinate - stride*idx)
    
def toPartBox(bbox, stride_w, stride_h, w_idx, h_idx):
    x0, y0, w, h = bbox
    x0 = transformCoordinate(x0, stride_w, w_idx)
    y0 = transformCoordinate(y0, stride_h, h_idx)
    return [x0, y0, w, h]
    
def splitImage(img_obj, image_root_dir, out_w, out_h, stride_w, stride_h):
    result = []
    original_img = img_obj.getOriginalImage(image_root_dir)
    img_w, img_h =  img_obj.getSize()
    # max number of images we can split in width
    no_img_w = (img_w - out_w) // stride_w + 1
    # max number of images we can split in height
    no_img_h = (img_h - out_h) // stride_h + 1
    splited_imgs = np.empty((no_img_w, no_img_h, out_h, out_w, 3), dtype=np.uint8)
    boxesWithClassID = img_obj.getBboxes()
    for i in range(no_img_w):
        for j in range(no_img_h):
            start_x = i*stride_w
            start_y = j*stride_h
            img_as_box = [start_x, start_y, start_x+out_w, start_y+out_h]
            splited_img = original_img[start_y:start_y+out_h, start_x:start_x+out_w]
            splited_imgs[i, j] = splited_img
            all_boxes = []
            for box, class_id in boxesWithClassID:
                absBox = cocoToAbsoluteBox(box)
                overlap_box = getOverlapBox(img_as_box, absBox)
                if len(overlap_box) != 0:
                    overlap_box_in_part = toPartBox(overlap_box, stride_w, stride_h, i, j)
                    all_boxes.append(Bbox(overlap_box_in_part, class_id))
            result.append(PartImageDetectionResult(splited_img, i, j, all_boxes))
    return result                    
    