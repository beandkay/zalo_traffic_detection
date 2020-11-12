import torch
import numpy as np
import json
import os
import cv2
import math
BLUE=(255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PINK = (255, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 127, 255)
CUSTOM = (255,170,170)
COLOR_CLASS = {0: BLUE, 1:GREEN, 2:RED, 3:YELLOW, 4:PINK, 5:BLACK, 6:ORANGE, 7:CUSTOM}
PIXEL = 1

def cocoToAbsoluteBox(cocoBox):
    return [int(cocoBox[0]), int(cocoBox[1]), int(cocoBox[0]+cocoBox[2]), int(cocoBox[1]+cocoBox[3])]
def toCocoBox(connerFormBox):
    w = connerFormBox[2]- connerFormBox[0]
    h = connerFormBox[3]- connerFormBox[1]
    return [connerFormBox[0], connerFormBox[1], w, h]

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

def non_max_suppression_fast(all_boxes, overlapThresh=0.1):
    # if there are no boxes, return an empty list
    boxes = [np.array(box_obj.toConnerForm(), dtype=np.float32).reshape(1, 4) for box_obj in all_boxes]
    if len(boxes) == 0:
        return []
    else:
        boxes = np.concatenate(boxes)
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by area of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

def transformCoordinate(coordinate, stride, idx, margin=0):
    return (coordinate - stride*idx + margin)

def transformToOriginalCoordinate(partCoordinate, stride, idx, margin=0):
    return stride*idx + partCoordinate + margin

def toOriginalBox(bbox,stride_w, stride_h, w_idx, h_idx, margin_x=0, margin_y=0):
    x0, y0, w, h = bbox
    x_new = transformToOriginalCoordinate(x0, stride_w, w_idx, margin_x)
    y_new = transformToOriginalCoordinate(y0, stride_h, h_idx, margin_y)
    return [x_new, y_new, w, h]

def toPartBox(bbox, stride_w, stride_h, w_idx, h_idx, margin_x=0, margin_y=0):
    x0, y0, w, h = bbox
    x0 = transformCoordinate(x0, stride_w, w_idx, margin_x)
    y0 = transformCoordinate(y0, stride_h, h_idx, margin_y)
    return [x0, y0, w, h]


def splitCvImage(original_img, out_w, out_h, stride_w, stride_h):
    result = []
    img_h, img_w =  original_img.shape[0], original_img.shape[1]
    # max number of images we can split in width
    no_img_w = (img_w - out_w) // stride_w + 1
    # max number of images we can split in height
    no_img_h = (img_h - out_h) // stride_h + 1
    splited_imgs = np.empty((no_img_w, no_img_h, out_h, out_w, 3), dtype=np.uint8)
    for i in range(no_img_w):
        for j in range(no_img_h):
            start_x = i*stride_w
            start_y = j*stride_h
            img_as_box = [start_x, start_y, start_x+out_w, start_y+out_h]
            splited_img = original_img[start_y:start_y+out_h, start_x:start_x+out_w]
            splited_imgs[i, j] = splited_img
    return splited_imgs

class PartImageDetectionResult:
    '''
    @param image: cropped image
    @param w_index: index of image along width (start from 0)
    @param h_index: index of image along height (start from 0)
    @bboxes Bbox: object hold bounding box and class id denoted cat
    '''
    def __init__(self, image, w_index, h_index, bboxes):
        self.image = image
        self.w_index = w_index
        self.h_index = h_index
        self.bboxes = bboxes
        self.out_h, self.out_w = self.image.shape[:2]

    def get_all_cats(self):
        all_cats = [box.cat for box in self.bboxes]
        if len(all_cats) > 0:
            return list(dict.fromkeys(all_cats))
        return []

    def get_number_box(self):
        number_box = 0
        for box in self.bboxes:
            number_box += 1
        return number_box

    def get_nummber_box_by_cat_id(self, cat_id):
        number_box = 0
        for box in self.bboxes:
            if box.cat == cat_id:
                number_box += 1
        return number_box

    def get_boxes_on_original_images(self, org_h, org_w, stride_w, stride_h):
        original_bboxes = []
        if len(self.bboxes) > 0:
            for box_obj in self.bboxes:
                box_coords = box_obj.getCocoBoxFormat()
                margin_x = 0
                margin_y = 0
                # print(self.w_index, stride_w, self.out_w, org_w)
                if (self.w_index * stride_w + self.out_w) > org_w:
                    margin_x = org_w - (self.w_index * stride_w + self.out_w)

                if (self.h_index * stride_h + self.out_h) > org_h:
                    margin_y = org_h - (self.h_index * stride_h + self.out_h)
                # print(margin_x, margin_y)
                new_box_coords = toOriginalBox(box_coords, stride_w, stride_h, self.w_index, self.h_index, margin_x, margin_y)
                cat = box_obj.cat
                new_box = Bbox(new_box_coords, cat)
                new_box.set_score(box_obj.get_score())
                original_bboxes.append(new_box)
        return original_bboxes

    def drawBboxOnImage(self):
        img = self.image.copy()
        for box in self.bboxes:
            img = box.onImage(img)
        return img

class ImageDetectionResult:
    def __init__(self, img_meta, images_dir, out_w, out_h, stride_w, stride_h):
        self.img_meta = img_meta
        self.images_dir = images_dir
        self.h = self.img_meta["height"]
        self.w = self.img_meta["width"]
        self.out_w = out_w
        self.out_h = out_h
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.detection_bboxes = []
        self.orginal_image = self._load_origin_image()
        # self.splited_imgs = self._split_imgs()
        self.part_img_objects = []


    def _load_origin_image(self):
        image_name = self.img_meta["file_name"]
        image_path = os.path.join(self.images_dir, image_name)
        if not os.path.exists(image_path):
            raise "does not exist {}".format(image_path)
        return cv2.imread(image_path)

    def _group_detections(self, number_classes, threshold):
        final_bboxes = []
        no_img_h = math.ceil((self.h - self.out_h)/self.stride_h) + 1
        no_img_w = math.ceil((self.w - self.out_w)/self.stride_w) + 1
        for class_id in range(number_classes):
            all_bboxes = []
            # no_img_w, no_img_h = self.splited_imgs.shape[:2]
            # print(no_img_w, no_img_h)
            count = 0
            for i in range(no_img_w):
                for j in range(no_img_h):
                    original_boxes = self.part_img_objects[i][j].get_boxes_on_original_images(self.h, self.w, self.stride_w, self.stride_h)
                    if len(original_boxes) > 0:
                        for box in original_boxes:
                            if box.cat == class_id:
                                box.set_part_index(i, j)
                                # check box ratio
                                ratio_w_h = box.box_ratio_w_h()
                                if ratio_w_h < 3 and ratio_w_h > 1/3:
                                    all_bboxes.append(box)
                        # all_bboxes += [box for box in original_boxes if box.cat == class_id]

            sorted(all_bboxes, key=lambda x: x.area(), reverse=True)
            pick = non_max_suppression_fast(all_bboxes, threshold)
            remain_boxes = [all_bboxes[pick_i] for pick_i in pick]
            final_bboxes += remain_boxes
        return final_bboxes
    def _group_detections_v2(self, number_classes, threshold):
        final_bboxes = []
        no_img_h = math.ceil((self.h - self.out_h)/self.stride_h) + 1
        no_img_w = math.ceil((self.w - self.out_w)/self.stride_w) + 1
        all_bboxes = []
        # no_img_w, no_img_h = self.splited_imgs.shape[:2]
        # print(no_img_w, no_img_h)
        count = 0
        for i in range(no_img_w):
            for j in range(no_img_h):
                original_boxes = self.part_img_objects[i][j].get_boxes_on_original_images(self.h, self.w, self.stride_w, self.stride_h)
                if len(original_boxes) > 0:
                    for box in original_boxes:
                        # if box.cat == class_id:
                        box.set_part_index(i, j)
                        all_bboxes.append(box)
                    # all_bboxes += [box for box in original_boxes if box.cat == class_id]

        sorted(all_bboxes, key=lambda x: x.area(), reverse=True)
        pick = non_max_suppression_fast(all_bboxes, threshold)
        final_bboxes = [all_bboxes[pick_i] for pick_i in pick]
            # final_bboxes += remain_boxes
        return final_bboxes

    def merge_boxes(self, detection_results, number_classes=7, threshold=0.1):
        self._load_detection_results(detection_results)
        remain_bboxes = self._group_detections(number_classes, threshold)
        return ImageAnnotation(self.img_meta["image_id"], self.h, self.w, self.img_meta["file_name"], remain_bboxes)

    def merge_boxes_v2(self, detection_results, number_classes=7, threshold=0.1):
        self._load_detection_results(detection_results)
        remain_bboxes = self._group_detections(number_classes, threshold)
        sorted(remain_bboxes, key=lambda x: x.get_score(), reverse=True)
        pick = non_max_suppression_fast(remain_bboxes, 0.4)
        final_bboxes = [remain_bboxes[pick_i] for pick_i in pick]
        return ImageAnnotation(self.img_meta["image_id"], self.h, self.w, self.img_meta["file_name"], final_bboxes)


    def _load_detection_results(self, detection_results):
        self.part_img_objects = detection_results


class ImageAnnotation:
    '''
    class holding all information of image annotation
    '''
    def __init__(self, image_id, h, w, file_name, bboxes):
        self.image_id = image_id
        self.h = h
        self.w = w
        self.file_name = file_name
        # bboxes include box and class_id, format [[x1, y1, w, h], class_id]
        self.bboxes = bboxes

    def get_image_meta(self):
        return {"image_id": self.image_id, "height": self.h, "width": self.w, "file_name": self.file_name}

    def getSize(self):
        return self.w, self.h

    def getBboxes(self):
        '''
        return bboxes with class_id
        '''
        return self.bboxes

    def getBBoxesWithoutClassId(self):
        '''
        return all boxes without class_id
        '''
        return [box.getCocoBoxFormat() for box in self.bboxes]

    def getFileName(self):
        return self.file_name

    def getOriginalImage(self, root_dir):
        image_path = os.path.join(root_dir, self.file_name)
        if not os.path.exists(image_path):
            print("file does not exist, return empty image")
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)
        return cv2.imread(image_path)

    def bboxesOnImage(self, root):
        '''
        draw bboxes on images for each box
        @param root : dataset directory
        '''
        image_origin = self.getOriginalImage(root)
        overlaid_img = image_origin.copy()
        for box in self.bboxes:
            overlaid_img = box.onImage(overlaid_img)
        return image_origin, overlaid_img

    def toJson(self):
        json_str = []
        for box in self.bboxes:
            report_dict = {}
            report_dict["image_id"] = self.image_id
            report_dict["bbox"] = box.getCocoBoxFormat()
            report_dict["score"] = box.get_score()
            report_dict["category_id"] = box.cat + 1
            json_str.append(report_dict)
        return json_str

    def __str__(self):
        report_str = "file_name: {}, image_id: {}".format(self.file_name, self.image_id)
        return report_str

class Bbox:
    def __init__(self, box, cat):
        '''
        @param box [x, y, w, h]
        @param cat class id
        '''
        self.box = box
        self.cat = cat
        self.w_index = -1
        self.h_index = -1
        self.score = 1.0
    def getCocoBoxFormat(self):
        return self.box

    def box_ratio_w_h(self):
        x, y, w, h = self.box
        return 1.0*w/h

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def get_box_center(self):
        if len(self.box) == 0:
            return 0, 0
        else:
            x, y, w, h = self.box
            return (x+w//2), int(y+h//2)

    def area(self):
        area = 0
        if len(self.box) != 0:
            area = self.box[2] * self.box[3]
        return area
    def set_part_index(self, w_index, h_index):
        self.w_index = w_index
        self.h_index = h_index

    def get_part_index(self):
        return self.w_index, self.h_index

    def toConnerForm(self):
        '''
        convert [x, y, w, h] -> [x1, y1, x2, y2]
        '''
        return cocoToAbsoluteBox(self.box)
    def __str__(self):
        return "bboxes x,y, w, h {} - cats {}".format(self.box, self.cat)

    def onImage(self, image):
        '''
        draw bbox on image
        '''
        start_point = (self.box[0], self.box[1])
        end_point = (self.box[0]+self.box[2], self.box[1]+self.box[3])
        overlaid_img = cv2.rectangle(image.copy(), start_point , end_point, COLOR_CLASS[self.cat], 2)
        return overlaid_img



def splitImageObj(img_obj, image_root_dir, out_w, out_h, stride_w, stride_h):
    result = []
    original_img = img_obj.getOriginalImage(image_root_dir)
    img_w, img_h =  img_obj.getSize()
    # max number of images we can split in width
    no_img_w = (img_w - out_w) // stride_w + 1
    # max number of images we can split in height
    no_img_h = (img_h - out_h) // stride_h + 1
    splited_imgs = np.empty((no_img_w, no_img_h, out_h, out_w, 3), dtype=np.uint8)
    boxesObj = img_obj.getBboxes()

    result = [[None for _ in range(no_img_h)] for _ in range(no_img_w)]
    for i in range(no_img_w):
        for j in range(no_img_h):
            start_x = i*stride_w
            start_y = j*stride_h
            img_as_box = [start_x, start_y, start_x+out_w, start_y+out_h]
            splited_img = original_img[start_y:start_y+out_h, start_x:start_x+out_w]
            splited_imgs[i, j] = splited_img
            all_boxes = []
            for box_obj in boxesObj:
                absBox = cocoToAbsoluteBox(box_obj.getCocoBoxFormat())
                overlap_box = getOverlapBox(img_as_box, absBox)
                if len(overlap_box) != 0:
                    overlap_box_in_part = toPartBox(overlap_box, stride_w, stride_h, i, j)
                    all_boxes.append(Bbox(overlap_box_in_part, box_obj.cat))
            result[i][j] = PartImageDetectionResult(splited_img, i, j, all_boxes)
    return result
