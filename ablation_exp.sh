
## simple clip model
cd "$(dirname "$0")"
pro_dir="./exp_log/ablation/simpleclip_epoch35_bs128_spaq"
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
                    --loss_type l1 \
                    --scene_sampling 0 \
                    --project_dir $pro_dir \
                    --train_dataset spaq \
                    --test_dataset koniq10k spaq livec live csiq bid cid2013 \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 

sleep 2m

## simple clip model, with our domain alignment
cd "$(dirname "$0")"
pro_dir="./exp_log/ablation/simpleclip_align_epoch35_bs128_spaq"
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
                    --loss_type scale_shift \
                    --scene_sampling 2 \
                    --project_dir $pro_dir \
                    --train_dataset spaq \
                    --test_dataset koniq10k spaq livec live csiq bid cid2013 \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 

sleep 2m

## use our local global complementary token combination, not domain alignment
cd "$(dirname "$0")"
pro_dir="./exp_log/ablation/our_notalign_epoch35_bs128_spaq"
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
                    --project_dir $pro_dir \
                    --train_dataset spaq \
                    --test_dataset koniq10k spaq livec live csiq bid cid2013 \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 

sleep 2m

## our full model
cd "$(dirname "$0")"
pro_dir="./exp_log/ablation/our_epoch35_bs128_spaq"
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
                    --test_dataset koniq10k spaq livec live csiq bid cid2013 \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 


# ## our full model, but all use cls for global and local
# cd "$(dirname "$0")"
# pro_dir="./exp_log/ablation/our_epoch35_bs128_spaq_allcls"
# mkdir -p $pro_dir
# clip_model="openai/ViT-B-16"

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#                     train_test_IQA.py  \
#                     --clip_model $clip_model \
#                     --epochs 35 \
#                     --lr 1e-5  \
#                     --warmup_epoch 5 \
#                     --weight_decay 1e-5 \
#                     --batch_size 64 \
#                     --local_global \
#                     --all_global \
#                     --loss_type scale_shift \
#                     --scene_sampling 2 \
#                     --project_dir $pro_dir \
#                     --train_dataset spaq \
#                     --test_dataset koniq10k spaq livec live csiq bid cid2013 \
#                     --exp_type cross-set \
#                     >> $pro_dir/train.log 
