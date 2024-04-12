cd "$(dirname "$0")"
pro_dir="./ablation/clip_base_epoch50_bs64"
if [ ! -d $pro_dir ]; then
    mkdir -p $pro_dir
fi
# clip_model="dfn2b/ViT-L-14-quickgelu"
clip_model="openai/ViT-B-16"

CUDA_VISIBLE_DEVICES=1 accelerate launch \
                    src/train_test_IQA.py  \
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
                    --train_data 0 \
                    > $pro_dir/train.log 