
import json 
from shutil import copyfile, move
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='fold ')
    args = parser.parse_args()

    print('train')
    with open('kfold_160/train_fold_{}.json'.format(args.fold)) as json_file:
        data = json.load(json_file)

    list_anno = {}
    w = 160
    h = 160
    for image in data['images']:

        list_anno[image['id']] = []


    for anno in data['annotations']:
        id_ = anno['image_id']
        bbox = anno['bbox']

        #xywh
        x_center = (bbox[0] + bbox[2]/2)/w
        y_center = (bbox[1] + bbox[3]/2)/h
        width = bbox[2]/w
        height = bbox[3]/h

        class_ = anno['category_id']
        
        # if id_ in list_anno:
        list_anno[id_].append([class_, x_center, y_center, width, height])

    roordir = 'data_fold{}/labels/train'.format(args.fold)
    for id_ in list_anno.keys():

        f= open(os.path.join(roordir, '{}.txt'.format(id_)),"w+")
        for anno in list_anno[id_]:

            f.write("{} {} {} {} {}\n".format(anno[0] - 1, anno[1], anno[2], anno[3], anno[4]))
        f.close()





    print('val')
    with open('kfold_160/val_fold_{}.json'.format(args.fold)) as json_file:
        data = json.load(json_file)

    list_anno = {}
    w = 160
    h = 160
    for image in data['images']:

        list_anno[image['id']] = []


    for anno in data['annotations']:
        id_ = anno['image_id']
        bbox = anno['bbox']

        #xywh
        x_center = (bbox[0] + bbox[2]/2)/w
        y_center = (bbox[1] + bbox[3]/2)/h
        width = bbox[2]/w
        height = bbox[3]/h

        class_ = anno['category_id']
        
        # if id_ in list_anno:
        list_anno[id_].append([class_, x_center, y_center, width, height])

    roordir = 'data_fold{}/labels/val'.format(args.fold)
    for id_ in list_anno.keys():

        f= open(os.path.join(roordir, '{}.txt'.format(id_)),"w+")
        for anno in list_anno[id_]:

            f.write("{} {} {} {} {}\n".format(anno[0] - 1, anno[1], anno[2], anno[3], anno[4]))
        f.close()




