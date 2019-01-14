python3 main.py -a resnet50 --lr 0.1 -b 256 --epochs 200 \
               --warm_up 5 --mixup 0.2 --label_smooth_eta 0.1 \
               imagenet2012-train.txt imagenet2012-val.txt