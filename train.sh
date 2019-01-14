python3 main.py -a resnet50 --lr 0.01 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 \
               -b 256 --warm_up 5 --mixup 0.2 --epochs 200 --label_smooth_eta 0.1
              [imagenet-folder with train and val folders]