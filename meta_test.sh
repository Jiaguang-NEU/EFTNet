#!/bin/sh
PARTITION=Segmentation

GPU_ID=0
dataset=pascal # pascal coco fss
exp_name=split0

arch=EFTNet
net=vgg # vgg resnet50 resnet101


now=$(date +"%Y-%m-%d_%X")
exp_dir=exp_ft_EFT/${dataset}/MSANet/${exp_name}/${net}
snapshot_dir=weights/${dataset}/${exp_name}/${net}
result_dir=${exp_dir}/result/EFTNet/${now}
show_dir=${exp_dir}/show
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
mkdir -p ${snapshot_dir} ${result_dir} ${show_dir}
cp meta_test.sh meta_test.py ${config} ${result_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u meta_test.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/test-$now.log