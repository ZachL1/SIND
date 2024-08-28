# /bin/bash

## TQA task
# following LIQE paper, we train and test ten times with different random splits
# we use the same random splits as the LIQE

# Joint training on mix datasets: koniq10k livec bid kadid10k csiq live
pro_dir="./exp_log/random-split/our_epoch35_bs128_all_dataset"
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
                    --loss_type scale_shift  \
                    --scene_sampling 2 \
                    --project_dir $pro_dir \
                    --train_dataset koniq10k livec bid kadid10k csiq live \
                    --test_dataset koniq10k livec bid kadid10k csiq live \
                    --exp_type random-split \
                    --dataset_domain \
                    >> $pro_dir/train.log 

## AQA task
# following AesCLIP paper
# we train and test ten times with different random splits of EVA dataset
pro_dir="./exp_log/random-split/our_epoch50_bs128_eva"
mkdir -p $pro_dir
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 50 \
                    --lr 1e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --local_global \
                    --loss_type scale_shift  \
                    --scene_sampling 2 \
                    --project_dir $pro_dir \
                    --train_dataset eva \
                    --test_dataset eva \
                    --exp_type random-split \
                    >> $pro_dir/train.log 

# we use official split of PARA dataset
cd "$(dirname "$0")"
pro_dir="./exp_log/random-split/our_epoch80_bs128_para"
mkdir -p $pro_dir
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 80 \
                    --lr 1e-5  \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --local_global \
                    --loss_type scale_shift  \
                    --scene_sampling 2 \
                    --project_dir $pro_dir \
                    --train_dataset para \
                    --test_dataset para \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 