
model_ckpt=exp_log_bak/random-split/our_epoch35_bs128_all_dataset/split_1/ckpts/model_epoch035.pth

for json_file in data_json/for_cross_set/test/*.json
# for json_file in data_json/all/*.json
do
    echo $json_file
    CUDA_VISIBLE_DEVICES=1 python test_IQA.py --batch_size 8 --data_dir ./data --json_file $json_file --model_path $model_ckpt
done

# CUDA_VISIBLE_DEVICES=1 python test_IQA.py --batch_size 8 --data_dir data/HUAWEI_test --model_path $model_ckpt

