import json
from ensemble_boxes import *
import argparse
import glob
import os
WIDTH = 1622
HEIGHT = 626

def getBoxes(data):
    box_with_image_id = {}
    for result in data:
        img_id = result["image_id"]
        detection = {"bbox": result["bbox"], "category_id": result["category_id"], "score":result["score"]}
        if not img_id in box_with_image_id.keys():
            box_with_image_id[img_id] = [detection]
        else:
            box_with_image_id[img_id].append(detection)
    return box_with_image_id

def xyxy2xywh(box):
    x,y,w,h = box
    return [int(x), int(y), int(x+w), int(y+h)]

def CocoBoxFromNormBox(normBox, width, height):
    x1, y1, x2, y2 = normBox
    return [int(x1*width+0.5), int(y1*height+0.5), int((x2-x1)*width+0.5), int((y2-y1)*height+0.5)]

def getNormBoxes(bboxes, image_id, width, height):
    all_boxes = bboxes[image_id]
    box = []
    class_id = []
    score = []
    for box_det in all_boxes:
        x1,y1,x2,y2 = xyxy2xywh(box_det['bbox'])
        box.append([x1/width, y1/height, x2/width, y2/height])
        class_id.append(box_det['category_id'])
        score.append(box_det['score'])
    return box, class_id, score
def toJson(image_id, boxes, class_ids, scores, width, height):
    result = []
    for i in range(len(class_ids)):
        json_str = {}
        json_str["image_id"] = image_id
        json_str["category_id"] = int(class_ids[i])
        json_str["score"] = float(scores[i])
        json_str["bbox"] = CocoBoxFromNormBox(boxes[i], width, height)
        result.append(json_str)
    return result

def load_all_dets(all_pred_jsons):
    all_dets = []
    weights = []
    for det_json in all_pred_jsons:
        if 'fold_1' in det_json:
            weights.append(1)
        elif 'fold_3' in det_json:
            weights.append(3)
        else:
            weights.append(2)

        with open(det_json, 'r') as f:
            prediction = json.load(f)
            all_dets.append(getBoxes(prediction))

    for i in range(len(weights)):
        print(all_pred_jsons[i], weights[i])
    return all_dets, weights


def wbf(all_dets, iou_thr=0.9, skip_box_thr=0.001, width=1622, height=626, weights = None):
    data = []
    image_ids = set().union(*all_dets)
    for image_id in image_ids:
        boxes_to_fusion = []
        class_to_fusion = []
        scores_to_fusion = []
        for single_fold_det in all_dets:
            if image_id in single_fold_det.keys():
                box, class_id, score = getNormBoxes(single_fold_det, image_id, width, height)
                boxes_to_fusion.append(box)
                class_to_fusion.append(class_id)
                scores_to_fusion.append(score)
        if len(boxes_to_fusion) >= 4:
            # boxes, scores, labels = weighted_boxes_fusion(boxes_to_fusion, scores_to_fusion, class_to_fusion, weights=[1, 2, 3, 2, 2], iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            boxes, scores, labels = weighted_boxes_fusion(boxes_to_fusion, scores_to_fusion, class_to_fusion, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        else:
            boxes = boxes_to_fusion[0]
            scores = scores_to_fusion[0]
            labels = class_to_fusion[0]
        json_str = toJson(image_id, boxes, labels, scores, WIDTH, HEIGHT)
        data += json_str
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, default='predict/json', help='directory store all json file from each model prediction')
    parser.add_argument('--output_name', type=str, default='/result/submission.json', help='final output json')
    
    args = parser.parse_args()
    json_wildcard = os.path.join(args.json_dir, "*.json")
    json_files = glob.glob(json_wildcard)
    all_dets, weights = load_all_dets(json_files)
    final_preds = wbf(all_dets, iou_thr=0.9, skip_box_thr=0.001, weights = weights)
    with open(args.output_name, 'w') as f:
        json.dump(final_preds, f)