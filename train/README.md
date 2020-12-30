# train 
```bash
python3 train.py --img 160 --batch <sô lưởng ảnh trong 1 batch size> --epochs 50 --data zalo_traffic_160_fold<1 to 5>.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml --hyp data/hyp.zalo.yaml --multi-scale --worker 20 
```
## ví dụ train với 2 card 2080ti fold1 
```bash
python3 train.py --img 160 --batch 160 --epochs 50 --data zalo_traffic_160_fold1.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml --hyp data/hyp.zalo.yaml --multi-scale --worker 20 

```

# log
```bash
log và model được lưu trong thư mục run/exp*
```