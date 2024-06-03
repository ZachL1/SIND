

cd "$(dirname "$0")"
pro_dir="./exp_log/cross-set/our_epoch35_bs128_spaq"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 35 \
                    --lr 1e-5  \
                    --lr_ratio 1 \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --mixed_precision no \
                    --loss_type scale_shift  \
                    --input_size 224  \
                    --project_dir $pro_dir \
                    --scene_sampling 2 \
                    --train_dataset spaq \
                    --test_dataset koniq10k spaq livec agiqa3k kadid10k live csiq bid cid2013 \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 



# cd "$(dirname "$0")"
# pro_dir="./exp_log/cross-set/our_epoch35_bs128_kadid_update"
# if [ ! -d $pro_dir ]; then
#     mkdir -p $pro_dir
# fi
# # clip_model="dfn2b/ViT-L-14-quickgelu"
# clip_model="openai/ViT-B-16"

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#                     train_test_IQA.py  \
#                     --clip_model $clip_model \
#                     --epochs 35 \
#                     --lr 1e-5  \
#                     --lr_ratio 1 \
#                     --warmup_epoch 5 \
#                     --weight_decay 1e-5 \
#                     --batch_size 64 \
#                     --mixed_precision no \
#                     --loss_type scale_shift  \
#                     --input_size 224  \
#                     --project_dir $pro_dir \
#                     --scene_sampling 2 \
#                     --train_dataset kadid10k \
#                     --test_dataset koniq10k spaq livec agiqa3k kadid10k live csiq \
#                     --exp_type cross-set \
#                     >> $pro_dir/train.log 


cd "$(dirname "$0")"
pro_dir="./exp_log/cross-set/our_epoch35_bs128_spaq_koniq"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 35 \
                    --lr 1e-5  \
                    --lr_ratio 1 \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --mixed_precision no \
                    --loss_type scale_shift  \
                    --input_size 224  \
                    --project_dir $pro_dir \
                    --scene_sampling 2 \
                    --train_dataset spaq koniq10k \
                    --test_dataset koniq10k spaq livec agiqa3k kadid10k live csiq \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 



cd "$(dirname "$0")"
pro_dir="./exp_log/cross-set/our_epoch35_bs128_spaq_koniq_kadid"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 35 \
                    --lr 1e-5  \
                    --lr_ratio 1 \
                    --warmup_epoch 5 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --mixed_precision no \
                    --loss_type scale_shift  \
                    --input_size 224  \
                    --project_dir $pro_dir \
                    --scene_sampling 2 \
                    --train_dataset spaq koniq10k kadid10k \
                    --test_dataset koniq10k spaq livec agiqa3k kadid10k live csiq \
                    --exp_type cross-set \
                    >> $pro_dir/train.log 


