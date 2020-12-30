import torch
import numpy as np
import json
import os
import cv2
import random

import matplotlib.pyplot as plt
# import seaborn as sns


BLUE=(255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PINK = (255, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 127, 255)
CUSTOM = (255,170,170)
COLOR_CLASS = {0: BLUE, 1:GREEN, 2:RED, 3:YELLOW, 4:PINK, 5:BLACK, 6:ORANGE, 7:CUSTOM}

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def cocoToAbsoluteBox(cocoBox):
    #xywh -> xyxy
    return [cocoBox[0], cocoBox[1], cocoBox[0]+cocoBox[2], cocoBox[1]+cocoBox[3]]



class ImageAnnotation:
    '''
    class holding all information of image annotation
    '''
    def __init__(self, image_id, h, w, file_name, bboxes, split_bbox = []):
        self.image_id = image_id
        self.h = h
        self.w = w
        self.file_name = file_name
        # bboxes include box and class_id, format [[x1, y1, w, h], class_id]
        self.bboxes = bboxes

    def getSize(self):
        return self.w, self.h

    def getBboxes(self):
        '''
        return bboxes with class_id
        '''
        return self.bboxes
    
    def getClassId(self):
        return [box[1] for box in self.bboxes]
    
    def getBBoxesWithoutClassId(self):
        '''
        return all boxes without class_id
        '''
        return [box[0] for box in self.bboxes]
        
    def getFileName(self):
        return self.file_name

    def bboxesOnImage(self, root, save_dir):
        '''
        draw bboxes on images for each box
        @param root : dataset directory
        @param save_dir : overlaid image directory
        '''
        image_path = os.path.join(root, self.file_name)
        image = cv2.imread(image_path)
        for box, class_id in self.bboxes:
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[0])+int(box[2]), int(box[1])+int(box[3]))
            image = cv2.rectangle(image, start_point , end_point, COLOR_CLASS[class_id], 1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_save_path = os.path.join(save_dir, self.file_name)
        # print(image)
        cv2.imwrite(img_save_path, image)
    
    def __str__(self):
        report_str = "image_name: {}, image_id: {}, bboxes with class: {}".format(self.file_name, self.image_id, self.bboxes)
        return report_str

def filter_bbox(list_bbox):
    black_list = []
    for i, bbox_class_id in enumerate(list_bbox):
        bbox, class_id = bbox_class_id
        for j in range(i+1, len(list_bbox)):
            bbox2, class_id_2 = list_bbox[j]
            if bb_intersection_over_union(cocoToAbsoluteBox(bbox), cocoToAbsoluteBox(bbox2)) > 0.3 :#and class_id == class_id_2:
            # if overlay(cocoToAbsoluteBox(bbox), cocoToAbsoluteBox(bbox2)) > 0.8 and class_id == class_id_2:
                black_list.append(j)

    black_list = sorted(list(set(black_list)))
    # print('black_list {}'.format(black_list))
    for i in black_list[::-1]:
        list_bbox.pop(i)
    return list_bbox

def merge_bbox(bbox_1, bbox_2, weight_1, weight_2):
    return [int((bbox_1[i]*weight_1 + bbox_2[i]*weight_2)/(weight_1 + weight_2)) for i in range(len(bbox_1))]

def filter_bbox2(list_bbox):
    black_list = []
 
    for i, bbox_class_id in enumerate(list_bbox):
        if i in black_list: continue
            
        bbox, class_id = bbox_class_id
        weight_1 = 1
        weight_2 = 1
        
        for j in range(i+1, len(list_bbox)):
            bbox2, class_id_2 = list_bbox[j]
            if bb_intersection_over_union(cocoToAbsoluteBox(bbox), cocoToAbsoluteBox(bbox2)) > 0.45 and class_id == class_id_2:

                black_list.append(j)
                
                bbox = merge_bbox(bbox, bbox2, weight_1, weight_2)
                weight_1 += 1
        
        list_bbox[i] = [bbox, class_id]
                
    black_list = sorted(list(set(black_list)))
    # print('black_list {}'.format(black_list))
    for i in black_list[::-1]:
        list_bbox.pop(i)
        
    return list_bbox


def create_json(data, split_images, list_id = [], json_dir = 'train_traffic_sign_dataset_split_image.json'):

    new_json = {'annotations': [], 'images': []}
    new_json['info'] = data['info']
    new_json['categories'] = data['categories']

    anno_id = 0
    for i, image_info in enumerate(split_images):
#         print(image_info.image_id)
        if list_id:
            if image_info.image_id in list_id:
#                 print(i)
                new_json['images'].append({'file_name': image_info.file_name,
                                          'height': image_info.h,
                                          'width': image_info.w,
                                          'id': image_info.image_id})

                for bbox_id in image_info.getBboxes():
                    bbox , category_id = bbox_id[0], bbox_id[1]

                    new_json['annotations'].append({'area': bbox[2]*bbox[3],
                                                   "iscrowd": 0,
                                                   'image_id': image_info.image_id,
                                                   'bbox': bbox,
                                                   'category_id': category_id,
                                                   'id': anno_id})
                    anno_id += 1
        else:
            new_json['images'].append({'file_name': image_info.file_name,
                                          'height': image_info.h,
                                          'width': image_info.w,
                                          'id': image_info.image_id})

            for bbox_id in image_info.getBboxes():
                bbox , category_id = bbox_id[0], bbox_id[1]

                new_json['annotations'].append({'area': bbox[2]*bbox[3],
                                               "iscrowd": 0,
                                               'image_id': image_info.image_id,
                                               'bbox': bbox,
                                               'category_id': category_id,
                                               'id': anno_id})
                anno_id += 1
                
    with open(json_dir, 'w') as outfile:
        json.dump(new_json, outfile)
    return new_json


import math
def create_new_bbox(split_bbox, bbox):
    xmin = bbox[0]
    xmax = bbox[0] + bbox[2]
    ymin = bbox[1]
    ymax = bbox[1] + bbox[3]

    if xmin < split_bbox[0]:
        xmin = split_bbox[0]
    if xmax > split_bbox[2]:
        xmax = split_bbox[2]
    if ymin < split_bbox[1]:
        ymin = split_bbox[1]
    if ymax > split_bbox[3]:
        ymax = split_bbox[3]
    
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def check_new_bbox(new_bbox, old_bbox):
    s_new = new_bbox[2] * new_bbox[3]

    s_old = old_bbox[2] * old_bbox[3]
    return s_new/s_old
    
def create_split_image(data,
                       stride = (222, 114), 
                       root_dir = 'za_traffic_2020/traffic_train/images_split',
                       save_image = False,
                       h = 512, w = 512,
                      ):
    count = 0
    all_split_images = []
    
    image_ids = [image['id'] for image in data['images']]
    annos = {img_id:[] for img_id in image_ids}
    for anno in data['annotations']:
        annos[anno["image_id"]].append(anno)

    for i, images_info in enumerate(data["images"]):
        if save_image:
            original_image = cv2.imread('za_traffic_2020/traffic_train/images/{}'.format(images_info['file_name']))
#         print(math.ceil((1622-w)/stride[0]) + 1)
#         print(math.ceil((626-h)/stride[1]) + 1)
        for i in range(math.ceil((1622-w)/stride[0]) + 1):
            for j in range(math.ceil((626-h)/stride[1]) + 1):

                img_id = images_info['id']
                split_img_id = '{}_{}_{}'.format(images_info['id'], i, j)                    
                split_img_file_name = '{}.png'.format(split_img_id)
                
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
                    ymin = 626 - h
                    ymax = 626
                    
                split_bbox = [xmin, ymin, xmax, ymax]
#                 print(split_bbox)
                if save_image:
                    split_image = original_image[split_bbox[1]:split_bbox[3], split_bbox[0]:split_bbox[2]]

                bboxes = []
                for anno_data in annos[img_id]:
                    if bb_intersection_over_union(split_bbox, cocoToAbsoluteBox(anno_data['bbox'])) > 0:
                        bbox = create_new_bbox(split_bbox, anno_data['bbox'])

                        bbox[0] -= split_bbox[0]
                        bbox[1] -= split_bbox[1]

    #                     start_point = (bbox[0], bbox[1])
    #                     end_point = (bbox[2] + bbox[0], bbox[3] + bbox[1])
    #                     split_image = cv2.rectangle(split_image, start_point , end_point, (255, 255, 255), 2)

#                         if check_new_bbox(bbox,  anno_data['bbox']) < 0.25: count += 1
                        category_id = anno_data["category_id"]
                        bboxes.append((bbox, category_id))

                all_split_images.append(ImageAnnotation(split_img_id , h, w, split_img_file_name, bboxes))
                if len(bboxes) > 0:
                    count += 1
                if save_image:
                    cv2.imwrite('{}/{}'.format(root_dir, split_img_file_name), split_image)
#         break
    print(count)
    return all_split_images
