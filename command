python -u detect_zalo.py --weights /home/asilla/sonnh/checkpoint/fold3/last40.pt \
--img-size 608 --conf-thres 0.3 --iou-thres 0.1 --device 0 \
--output_dir /home/asilla/sonnh/test/test_fold_3_model_160_best_input_384_resize_608 \
--stride_w 84 --stride_h 84 --input_w 384 --input_h 384 \
--root_dir /home/asilla/sonnh/za_traffic_2020/traffic_public_test/images \
--save_image --json_dir /home/asilla/sonnh/test/test_fold_3_model_160_best_input_384_resize_608.json



python -u detect_zalo.py --weights /home/asilla/sonnh/checkpoint/fold2/best.pt \
--img-size 608 --conf-thres 0.3 --iou-thres 0.1 --device 0 \
--output_dir /home/asilla/sonnh/eval/val_fold_2_model_160_best_input_384_resize_608 \
--stride_w 84 --stride_h 84 --input_w 384 --input_h 384 --root_dir /home/asilla/sonnh/eval/val_fold2 \
--save_image --json_dir /home/asilla/sonnh/eval/val_fold_2_model_160_best_input_384_resize_608.json




python -u detect_zalo.py --weights /home/asilla/sonnh/checkpoint/fold2/model_160_last36.pt \
--img-size 608 --conf-thres 0.3 --iou-thres 0.1 --device 0 \
--output_dir /home/asilla/sonnh/test/test \
--stride_w 84 --stride_h 84 --input_w 384 --input_h 384 \
--root_dir /home/asilla/sonnh/za_traffic_2020/traffic_public_test/images \
--json_dir /home/asilla/sonnh/test/test.json