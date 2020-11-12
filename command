phase1(exp6): python train.py --img 512 --batch 32 --epochs 100 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml
phase2(exp15): python train.py --img 512 --batch 32 --epochs 100 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml\
--hyp data/hyp.phase2.yaml --multi-scale

phase3(exp25): python train.py --img 384 --batch 32 --epochs 100 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml\
--hyp data/hyp.phase2.yaml --multi-scale --worker 16

phase4(exp25): python train.py --img 384 --batch 32 --epochs 100 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml --hyp data/hyp.phase2.yaml --multi-scale --worker 16 --resume --weights runs/exp25/weights/last.pt> logs/log_phase4.log &

phase5(exp40): python train.py --img 320 --batch 48 --epochs 50 --data zalo_traffic.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml --hyp data/hyp.phase5.yaml --multi-scale --worker 16


phase6(exp47): python train.py --img 160 --batch 128 --epochs 50 --data zalo_traffic_160.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml --hyp data/hyp.phase6.yaml --multi-scale --worker 20
phase7(exp47): python train.py --img 160 --batch 176 --epochs 50 --data zalo_traffic_160.yaml --weights '' --cfg yolov5x_zalo_traffic.yaml --hyp data/hyp.phase6.yaml --multi-scale --worker 20 --weights runs/exp47/weights/last.pt --resume