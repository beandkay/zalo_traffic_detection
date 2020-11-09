phase1(exp6): python train.py --img 512 --batch 32 --epochs 100 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml
phase2(exp15): python train.py --img 512 --batch 32 --epochs 100 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml\
 --hyp data/hyp.phase2.yaml --multi-scale

phase3(exp25): python train.py --img 384 --batch 32 --epochs 100 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml\
 --hyp data/hyp.phase2.yaml --multi-scale --worker 16