#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

# ========greenhouse dataset==========
python main.py --config configs/config.gh_dataset.unet_resnet152_pretrained.json --master_port 1220 --max_epoch 300 -b 4
python main.py --config configs/config.gh_dataset.unet_resnet152_pretrained.json --mode eval --eval_batch_size 4 --master_port 1221
python main.py --config configs/config.gh_dataset.unet_resnet152_pretrained.json --mode eval_coco --eval_batch_size 4 --master_port 1222
python main.py --run_name gh_dataset.unet_resnet152_pretrained_bufferWidth_2 --in_filepath {in_filepath}/images/ --out_dirpath {out_dirpath}/results
