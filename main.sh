#!/bin/sh
SCRIPT_PATH=$(cd $(dirname $0);pwd)
cd ${SCRIPT_PATH}
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
pip install toml
echo "$model_save_dir"

python -u train.py  -c ./config/unet_CDCN_adam_lr1e-3.yaml  --toml "train.lr=0.01"  --toml "train.num_epochs=20"   --toml "train.warm_up_ratio=0.01"    --toml "train.batch_size=200"   --toml "test.batch_size=200"  --toml "model.header_type='patch_pixel'" --toml 'model.backbone="lcnet_small"'  2>&1 | tee  ../nohup_v0416.log  

# python -u train.py  -c ./config/unet_CDCN_adam_lr1e-3.yaml  --toml "train.lr=0.001"  --toml "train.num_epochs=20"   --toml "train.warm_up_ratio=0.05"    --toml "train.batch_size=200"   --toml "test.batch_size=200"  --toml "model.header_type='patch_pixel'" --toml 'model.backbone="lcnet_small"'  2>&1 | tee  ../nohup_v0416.log  