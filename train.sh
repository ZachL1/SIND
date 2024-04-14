cd "$(dirname "$0")"
pro_dir="./exp_log/clip_base_epoch50_bs128_piq23"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 50 \
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
                    --train_dataset piq23 \
                    --test_dataset piq23 \
                    >> $pro_dir/train.log 



cd "$(dirname "$0")"
pro_dir="./exp_log/clip_glob_epoch50_bs128_piq23"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 50 \
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
                    --train_dataset piq23 \
                    --test_dataset piq23 \
                    --all_global \
                    >> $pro_dir/train.log 




cd "$(dirname "$0")"
pro_dir="./exp_log/clip_base_epoch50_bs128_spaq"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 50 \
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
                    --test_dataset spaq \
                    >> $pro_dir/train.log 


cd "$(dirname "$0")"
pro_dir="./exp_log/clip_glob_epoch50_bs128_spaq"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 50 \
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
                    --test_dataset spaq \
                    --all_global \
                    >> $pro_dir/train.log 



cd "$(dirname "$0")"
pro_dir="./exp_log/clip_base_epoch50_bs128_piq23_wu3"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                    train_test_IQA.py  \
                    --clip_model $clip_model \
                    --epochs 50 \
                    --lr 1e-5  \
                    --lr_ratio 1 \
                    --warmup_epoch 3 \
                    --weight_decay 1e-5 \
                    --batch_size 64 \
                    --mixed_precision no \
                    --loss_type scale_shift  \
                    --input_size 224  \
                    --project_dir $pro_dir \
                    --scene_sampling 2 \
                    --train_dataset piq23 \
                    --test_dataset piq23 \
                    >> $pro_dir/train.log 