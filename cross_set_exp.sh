

## training on spaq and koniq10k training set, testing on koniq10k, spaq, livec, live, csiq, bid, cid2013
## use dataset-domain alignment
cd "$(dirname "$0")"
pro_dir="./exp_log/cross-set/our_epoch35_bs128_spaq_koniq_dataset"
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
                    --train_dataset spaq koniq10k \
                    --test_dataset koniq10k spaq livec live csiq bid cid2013 \
                    --exp_type cross-set \
                    --dataset_domain \
                    >> $pro_dir/train.log 


# ## training on spaq and koniq10k training set, testing on koniq10k, spaq, livec, live, csiq, bid, cid2013
# ## use scene-domain alignment
# cd "$(dirname "$0")"
# pro_dir="./exp_log/cross-set/our_epoch35_bs128_spaq_koniq"
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
#                     --loss_type scale_shift  \
#                     --scene_sampling 2 \
#                     --project_dir $pro_dir \
#                     --train_dataset spaq koniq10k \
#                     --test_dataset koniq10k spaq livec live csiq bid cid2013 \
#                     --exp_type cross-set \
#                     >> $pro_dir/train.log 


## training on spaq training set, testing on koniq10k, spaq, livec, live, csiq, bid, cid2013
## use scene-domain alignment
cd "$(dirname "$0")"
pro_dir="./exp_log/cross-set/our_epoch35_bs128_spaq"
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
                    --train_dataset spaq \
                    --test_dataset koniq10k spaq livec live csiq bid cid2013 \
                    --exp_type cross-set \
                    --dataset_domain \
                    >> $pro_dir/train.log 


## if you want to train on single GPU, this is an example
export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1

cd "$(dirname "$0")"
pro_dir="./exp_log/cross-set/our_epoch35_bs64_spaq_koniq_dataset"
mkdir -p $pro_dir
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
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
                    --train_dataset spaq koniq10k \
                    --test_dataset koniq10k spaq livec live csiq bid cid2013 \
                    --exp_type cross-set \
                    --dataset_domain \
                    >> $pro_dir/train.log 