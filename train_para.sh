

cd "$(dirname "$0")"
pro_dir="./exp_log/clip_our_epoch35_bs128_eva"
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
                    --train_dataset eva \
                    --test_dataset eva \
                    >> $pro_dir/train.log 



cd "$(dirname "$0")"
pro_dir="./exp_log/clip_our_epoch35_bs128_para"
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
                    --train_dataset para \
                    --test_dataset para \
                    >> $pro_dir/train.log 



cd "$(dirname "$0")"
pro_dir="./exp_log/clip_l1_epoch35_bs128_eva"
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
                    --loss_type l1  \
                    --input_size 224  \
                    --project_dir $pro_dir \
                    --scene_sampling 0 \
                    --train_dataset eva \
                    --test_dataset eva \
                    >> $pro_dir/train.log 



cd "$(dirname "$0")"
pro_dir="./exp_log/clip_l1_epoch35_bs128_para"
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
                    --loss_type l1  \
                    --input_size 224  \
                    --project_dir $pro_dir \
                    --scene_sampling 0 \
                    --train_dataset para \
                    --test_dataset para \
                    >> $pro_dir/train.log 


cd "$(dirname "$0")"
pro_dir="./exp_log/clip_glob_epoch35_bs128_para"
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
                    --train_dataset para \
                    --test_dataset para \
                    --all_global \
                    >> $pro_dir/train.log 


cd "$(dirname "$0")"
pro_dir="./exp_log/clip_glob_epoch35_bs128_eva"
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
                    --train_dataset eva \
                    --test_dataset eva \
                    --all_global \
                    >> $pro_dir/train.log 
