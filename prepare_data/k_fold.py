
from utils import * 



with open('za_traffic_2020/traffic_train/train_traffic_sign_dataset_merge_overlay_box.json') as json_file:
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


all_class = [(img_anno.image_id, img_anno.getClassId()) for img_anno in  all_images]

x = {}
y = {}
for i in range(len(all_class)):
    class_id  = '_'.join([str(j) for j in sorted(list(set(all_class[i][1])))])
    if class_id not in x:
        x[class_id] = [all_class[i][0]]
        y[class_id] = 1
    else:
        x[class_id].append(all_class[i][0])
        y[class_id] += 1
print(len(x), len(y))


x['multi'] = []
y['multi'] = 0  

keys = list(x.keys())
for key in keys:
    if y[key]  < 5: 
        x['multi'] += x[key]
        y['multi'] += y[key]
        del x[key]
        del y[key]
print(len(x), len(y))

X = []
Y = []
for key in x.keys():
    X += x[key]
    Y += [key]*y[key]
X = np.array(X)
Y = np.array(Y)



import numpy as np
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, Y)

fold = 1
for train_index, val_index in skf.split(X, Y):
    
    X_train_fold = list(X[train_index])
    X_val_fold = list(X[val_index])
#     print(X_fold)
    create_json(data, all_images, list_id = list(X_train_fold), json_dir = 'kfold/train_fold_{}.json'.format(fold))
    # create_json(data, all_images, list_id = list(X_val_fold), json_dir = 'za_traffic_2020/annotations/original_new/val_fold_{}.json'.format(fold))
    create_json(data, all_images, list_id = list(X_val_fold), json_dir = 'kfold/val_fold_{}.json'.format(fold))
    fold += 1
