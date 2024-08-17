# /bin/bash

# following LIQE paper, we train and test ten times with different random splits
# we use the same random splits as the LIQE
# test dataset: koniq10k livec live csiq bid cid2013
test_datasets=("koniq10k" "livec" "live" "csiq" "bid" "cid2013")

for dataset in ${test_datasets[@]}
do
  echo "Test on $dataset"
  pro_dir="./exp_log/random-split/our_epoch35_bs128_$dataset"
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
                      --train_dataset $dataset \
                      --test_dataset $dataset \
                      --exp_type random-split \
                      --dataset_domain \
                      >> $pro_dir/train.log 
  
  sleep 2m
done

