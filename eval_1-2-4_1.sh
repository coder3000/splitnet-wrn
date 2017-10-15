#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
checkpoint="./split_1-2-4_1/model.ckpt-119999"
basemodel="./group_1-2-4_1/model.ckpt-199999"
output_file="./split_1-2-4_1/eval-119999.pkl"
#data_dir="./cifar100/train_val_split"
data_dir="/data1/dalgu/cifar100/train_val_split"

python eval.py --checkpoint $checkpoint \
    --basemodel $basemodel \
    --output_file $output_file \
    --data_dir $data_dir \
    --batch_size 100 \
    --test_iter 100 \
    --num_residual_units 2 \
    --k 8 \
    --ngroups1 4 \
    --ngroups2 2 \
    --ngroups3 1 \
    --gpu_fraction 0.96 \
    --display 10 \
    #--finetune True \
    #--load_last_layers True \
