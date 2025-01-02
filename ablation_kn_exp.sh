#!/bin/bash

# Define all combinations
# batch size = 32, 64, 128
declare -A combinations=(
    ["1,32"]=1  ["1,64"]=1
    ["2,16"]=1  ["2,32"]=1 ["2,64"]=1
    ["4,8"]=1  ["4,16"]=1 ["4,32"]=1
    ["8,4"]=1  ["8,8"]=1 ["8,16"]=1
)

# batch size = 1, 2, 4, 8, 16, 256, 512
declare -A combinations=(
    ["1,1"]=1  ["1,2"]=1  ["1,4"]=1  ["1,8"]=1  ["1,16"]=1
    ["2,1"]=1  ["2,2"]=1  ["2,4"]=1  ["2,8"]=1
    ["4,1"]=1  ["4,2"]=1  ["4,4"]=1  ["4,64"]=1
    ["8,1"]=1  ["8,2"]=1  ["8,32"]=1 ["8,64"]=1
)

run_experiment() {
    local k=$1
    local n=$2
    local gpu_list=$3
    local total_bs=$((k * n))
    
    # Calculate required GPUs and adjusted batch size
    local required_gpus=$(( (total_bs + 63) / 64 ))
    local bs_per_gpu=$(( (total_bs + required_gpus - 1) / required_gpus ))
    local k_per_gpu=$(( (k + required_gpus - 1) / required_gpus ))
    
    pro_dir="./exp_log/kn/k${k}_n${n}_bs${total_bs}_spaq_koniq_dataset"
    mkdir -p $pro_dir
    
    export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1
    CUDA_VISIBLE_DEVICES=$gpu_list accelerate launch \
        train_test_IQA.py \
        --clip_model "openai/ViT-B-16" \
        --epochs 15 \
        --lr 1e-5 \
        --warmup_epoch 5 \
        --weight_decay 1e-5 \
        --batch_size $bs_per_gpu \
        --local_global \
        --loss_type scale_shift \
        --scene_sampling $k_per_gpu \
        --project_dir $pro_dir \
        --train_dataset spaq \
        --test_dataset koniq10k \
        --exp_type cross-set \
        --mixed_precision fp16 \
        >> $pro_dir/train.log &
    
    sleep 2m
    pid=$!

    echo "Started experiment k=$k n=$n on GPUs $gpu_list (PID: $pid)"
    echo "Total batch size: $total_bs, Per GPU batch size: $bs_per_gpu"
    return $pid
}

# Track available GPUs (total 8 GPUs)
# declare -A gpu_status
# for i in {0..1}; do
#     gpu_status[$i]=0
# done

find_free_gpus() {
    local needed=$1
    local found=0
    local gpu_list=""
    
    # Try to find enough free GPUs
    for i in {0..1}; do
        # 检查GPU是否被占用 (使用率>10% 或显存使用>500MB视为占用)
        # gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i)
        mem_util=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        
        if [ $mem_util -lt 500 ]; then
            if [ -z "$gpu_list" ]; then
                gpu_list="$i"
            else
                gpu_list="$gpu_list,$i"
            fi
            ((found++))
            
            if [ $found -eq $needed ]; then
                echo "$gpu_list"
                return 0
            fi
        fi
    done
    echo ""
    return 1
}

# Main execution loop
for k in 1 2 4 8; do
    for n in 1 2 4 8 16 32 64; do
        if ! [ ${combinations["$k,$n"]+_} ]; then
            continue
        fi

        total_bs=$((k * n))
        required_gpus=$(( (total_bs + 63) / 64 ))

        # Wait for enough free GPUs
        while true; do
            gpu_list=$(find_free_gpus $required_gpus)
            if [ ! -z "$gpu_list" ]; then
                break
            fi
            sleep 1m
        done

        # # Mark GPUs as used
        # IFS=',' read -ra GPUS <<< "$gpu_list"
        # for gpu in "${GPUS[@]}"; do
        #     gpu_status[$gpu]=1
        # done

        # Run experiment
        run_experiment $k $n "$gpu_list"
        pid=$!

        # # Monitor process and free GPUs when done
        # {
        #     wait $pid
        #     IFS=',' read -ra GPUS <<< "$gpu_list"
        #     for gpu in "${GPUS[@]}"; do
        #         gpu_status[$gpu]=0
        #     done
        # } &
        
        sleep 2
    done
done

wait
echo "All experiments completed!"