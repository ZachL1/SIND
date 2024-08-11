# /bin/bash

## train-test on SPAQ dataset, leave-one-out setting
cd "$(dirname "$0")"
pro_dir="./exp_log/leave_one_out/our_epoch35_bs128_spaq"
mkdir -p $pro_dir
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 35 \
                    --lr 1e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --local_global \
                    --loss_type scale_shift \
                    --scene_sampling 2 \
                    --project_dir $pro_dir \
                    --train_dataset spaq \
                    --test_dataset spaq \
                    --exp_type leave-one-out \
                    >> $pro_dir/train.log 

## for other datasets koniq10k, eva, para 
## same as above, just change the dataset name