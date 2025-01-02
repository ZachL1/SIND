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




## Model-agnostic Framework Verification
export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1

###################### vit #####################
cd "$(dirname "$0")"
pro_dir="./exp_log/leave_one_out/simpleclip_align_epoch20_bs128_eva"
mkdir -p $pro_dir
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 20 \
                    --lr 1e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 128 \
                    --loss_type scale_shift \
                    --scene_sampling 4 \
                    --project_dir $pro_dir \
                    --train_dataset eva \
                    --test_dataset eva \
                    --exp_type leave-one-out \
                    >> $pro_dir/train.log


cd "$(dirname "$0")"
pro_dir="./exp_log/leave_one_out/simpleclip_epoch15_bs128_eva"
mkdir -p $pro_dir
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 15 \
                    --lr 1e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 128 \
                    --loss_type l1 \
                    --scene_sampling 0 \
                    --project_dir $pro_dir \
                    --train_dataset eva \
                    --test_dataset eva \
                    --exp_type leave-one-out \
                    >> $pro_dir/train.log 



###################### resnet50 #####################
cd "$(dirname "$0")"
pro_dir="./exp_log/leave_one_out/resnet50_align_epoch15_bs128_spaq_lr2"
mkdir -p $pro_dir
# clip_model="openai/ViT-B-16"
clip_model="resnet50"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 15 \
                    --lr 4e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 128 \
                    --loss_type scale_shift \
                    --scene_sampling 4 \
                    --project_dir $pro_dir \
                    --train_dataset spaq \
                    --test_dataset spaq \
                    --exp_type leave-one-out \
                    >> $pro_dir/train.log &


cd "$(dirname "$0")"
pro_dir="./exp_log/leave_one_out/resnet50_epoch15_bs128_spaq_lr2"
mkdir -p $pro_dir
clip_model="resnet50"

CUDA_VISIBLE_DEVICES=1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 15 \
                    --lr 4e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 128 \
                    --loss_type l1 \
                    --scene_sampling 0 \
                    --project_dir $pro_dir \
                    --train_dataset spaq \
                    --test_dataset spaq \
                    --exp_type leave-one-out \
                    >> $pro_dir/train.log 



## Ablation
# use our local global complementary token combination, not domain alignment
cd "$(dirname "$0")"
pro_dir="./exp_log/leave_one_out/our_notalign_epoch15_bs128_spaq"
mkdir -p $pro_dir
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 15 \
                    --lr 1e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --local_global \
                    --loss_type l1 \
                    --scene_sampling 0 \
                    --project_dir $pro_dir \
                    --train_dataset spaq \
                    --test_dataset spaq \
                    --exp_type leave-one-out \
                    >> $pro_dir/train.log 