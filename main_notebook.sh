#!/bin/sh
SCRIPT_PATH=$(cd $(dirname $0);pwd)
cd ${SCRIPT_PATH}
# CUDA_VISIBLE_DEVICES=0,1,2,3
pip install toml
echo "$pwd"
python -u train.py  -c ./config/unet_CDCN_adam_lr1e-3.yaml --toml "train.lr=0.0001"  --toml "train.num_epochs=20"   --toml "train.warm_up_ratio=0.2"    --toml "train.batch_size=64"   --toml "test.batch_size=64"  --toml "model.header_type='patch_pixel'" --toml 'model.backbone="lcnet_small"'  2>&1 | tee  ../nohup_v0416.log  