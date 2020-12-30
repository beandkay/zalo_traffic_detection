
from utils import * 

with open('za_traffic_2020/traffic_train/train_traffic_sign_dataset.json') as json_file:
    data = json.load(json_file)
    
# print(data.keys())
# print(data['annotations'][0])
# print(data['images'][0])


image_ids = [image['id'] for image in data['images']]
data['annotations'][3]
annos = {img_id:[] for img_id in image_ids}

for anno in data['annotations']:
    annos[anno["image_id"]].append(anno)


all_images = [] # list of img annotation class
for i,images_info in enumerate(data["images"]):
    img_id = images_info['id']
    h, w , file_name = images_info['height'], images_info['width'], images_info['file_name']
    bboxes = []
    for anno_data in annos[img_id]:
        bbox = anno_data['bbox']
        category_id = anno_data["category_id"]
        bboxes.append((bbox, category_id))
    all_images.append(ImageAnnotation(img_id, h, w, file_name, bboxes))
count_boxes = sum([len(img_anno.getBBoxesWithoutClassId()) for img_anno in  all_images])
print('before merge ', count_boxes)


for image in all_images:
    image.bboxes = filter_bbox2(image.bboxes)


count_boxes = sum([len(img_anno.getBBoxesWithoutClassId()) for img_anno in  all_images])
print('after merge ', count_boxes)


create_json(data, all_images, json_dir = 'za_traffic_2020/traffic_train/train_traffic_sign_dataset_merge_overlay_box.json')