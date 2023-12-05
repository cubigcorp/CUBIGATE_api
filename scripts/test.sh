#!/bin/bash

set_data(){
    mkdir -p temp/$1
    cp /home/minsy/CUBIG/dp/result/chest/normal/lora/$1/*.png temp/$1
    rm -rf temp/$1/visualize*
    python preprocess.py --data_dir temp/$1 --class_name NORMAL
    cp /home/minsy/CUBIG/dp/result/chest/pneumonia/lora/$1/*.png temp/$1
    rm -rf temp/$1/visualize*
    python preprocess.py --data_dir temp/$1 --class_name PNEUMONIA
    mkdir -p temp/$1/train temp/$1/test
    python preprocess.py --data_dir temp/$1 --split
}

dp_test_dp(){
    python classify.py\
    --data_dir temp/$1 \
    --num_classes 2 \
    --suffix chest_{$1}_dp_test_dp \
    --device 0 \
    --train \
    --test
}
dp_test_non_dp(){
    python classify.py \
        --data_dir /home/minsy/CUBIG/dp/data/chest \
        --num_classes 2 \
        --checkpoint /home/minsy/CUBIG/checkpoints/chest_{$1}_dp_test_dp_model.pt \
        --suffix chest_{$1}_dp_test_non_dp \
        --device 0 \
        --test \
        --limit 100
}
non_dp_test_dp(){
    python classify.py \
        --data_dir temp/$1 \
        --num_classes 2 \
        --checkpoint /home/minsy/CUBIG/checkpoints/chest_non_dp_model.pt \
        --suffix chest_{$1}_non_dp_test_dp \
        --device 0 \
        --test
    python classify.py \
        --data_dir temp/$1 \
        --num_classes 2 \
        --checkpoint /home/minsy/CUBIG/checkpoints/chest_non_dp_100_test_dp_model.pt \
        --suffix chest_{$1}_non_dp_test_dp_100 \
        --device 0 \
        --test
}

for i in {0..17..5}
do
    # train/test dataset directory
    set_data $i
    # train dp & test dp
    dp_test_dp $i
    #  train dp & test non_dp
    dp_test_non_dp $i
    # train non_dp & test dp
    non_dp_test_dp $i
    rm -rf temp
done


# train/test dataset directory
    set_data 17
    # train dp & test dp
    dp_test_dp 17
    #  train dp & test non_dp
    dp_test_non_dp 17
    # train non_dp & test dp
    non_dp_test_dp 17
    rm -rf temp