from utils import *
import random
from shutil import copy
import argparse
random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='fold ')
    args = parser.parse_args()

   
    print('train')
    with open('kfold/train_fold_{}.json'.format(args.fold)) as json_file:
        data_fold = json.load(json_file)

    split_images_fold = create_split_image(data_fold, stride = (40, 40), h= 160, w= 160, 
                                        save_image = False)

    split_images_fold_positive = []
    split_images_fold_negative = []
    split_images_fold_negative_after_cut = []

    for image in split_images_fold:
        
        if image.bboxes: split_images_fold_positive.append(image)
        else:
            split_images_fold_negative.append(image)
            
    print('positive {}'.format(len(split_images_fold_positive)))
    print('negative {}'.format(len(split_images_fold_negative)))
            
    
    random.shuffle(split_images_fold_negative)

    for i in range(len(split_images_fold_positive)*4):
        split_images_fold_negative_after_cut.append(split_images_fold_negative[i])

    # split_images_fold_negative_after_cut = split_images_fold_negative
    print('negative after cut random {}'.format(len(split_images_fold_negative_after_cut)))

    for image in split_images_fold_positive:
        src = 'split_image/{}'.format(image.file_name)
        dst = 'data_fold{}/images/train/{}'.format(args.fold, image.file_name)
        copy(src, dst)
        
        
    for image in split_images_fold_negative_after_cut:
        src = 'split_image/{}'.format(image.file_name)
        dst = 'data_fold{}/images/train/{}'.format(args.fold, image.file_name)
        copy(src, dst)

    split_images_fold = split_images_fold_positive + split_images_fold_negative_after_cut
    len(split_images_fold)


    new_json = create_json(data_fold, 
                       split_images_fold,
                       json_dir = 'kfold_160/train_fold_{}.json'.format(args.fold))






    print('val')
    with open('kfold/val_fold_{}.json'.format(args.fold)) as json_file:
        data_fold = json.load(json_file)

    split_images_fold = create_split_image(data_fold, stride = (40, 40), h= 160, w= 160, 
                                        save_image = False)

    split_images_fold_positive = []
    split_images_fold_negative = []
    split_images_fold_negative_after_cut = []

    for image in split_images_fold:
        
        if image.bboxes: split_images_fold_positive.append(image)
        else:
            split_images_fold_negative.append(image)
            
    print('positive {}'.format(len(split_images_fold_positive)))
    print('negative {}'.format(len(split_images_fold_negative)))
            
    
    random.shuffle(split_images_fold_negative)

    for i in range(len(split_images_fold_positive)*4):
        split_images_fold_negative_after_cut.append(split_images_fold_negative[i])

    split_images_fold_negative_after_cut = split_images_fold_negative
    print('negative after cut random {}'.format(len(split_images_fold_negative_after_cut)))

    for image in split_images_fold_positive:
        src = 'split_image/{}'.format(image.file_name)
        dst = 'data_fold{}/images/val/{}'.format(args.fold, image.file_name)
        copy(src, dst)
        
        
    for image in split_images_fold_negative_after_cut:
        src = 'split_image/{}'.format(image.file_name)
        dst = 'data_fold{}/images/val/{}'.format(args.fold, image.file_name)
        copy(src, dst)

    split_images_fold = split_images_fold_positive + split_images_fold_negative_after_cut
    len(split_images_fold)


    new_json = create_json(data_fold, 
                       split_images_fold,
                       json_dir = 'kfold_160/val_fold_{}.json'.format(args.fold))