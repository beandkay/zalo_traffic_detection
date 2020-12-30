from utils import * 



with open('za_traffic_2020/traffic_train/train_traffic_sign_dataset_merge_overlay_box.json') as json_file:
    data_fold = json.load(json_file)
    
split_images_fold = create_split_image(data_fold, stride = (40, 40), h= 160, w= 160, 
                                      save_image = True, root_dir = 'split_image')



