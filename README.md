# zalo_traffic_detection

fork from [YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v3.1)

# Run each fold predictions
```bash
mkdir predictions
python3 zalo_det_v3.py  --weights ./checkpoint/fold1.pt <1 to 5>  --img-size 640 --conf-thres 0.3 --iou-thres 0.1 --device 0 --stride_w 84 --stride_h 84 --input_w 384 --input_h 384 --root_dir <path to private test images ./private_test/images> --json_dir predictions/fold1.json <test_fold<1 to 5>.json>
```
# Run wbf to get final predictions
```bash
python3 wbf.py --json_dir predictions --output_name final_pred.json
```
> submit the final_pred.json file to get private test score
