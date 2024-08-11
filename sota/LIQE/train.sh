# CUDA_VISIBLE_DEVICES=0 python -u train_liqe_single.py > exp_log/leave_one_out/koniq10k/train.log

# CUDA_VISIBLE_DEVICES=1 python -u train_liqe_single.py > exp_log/leave_one_out/spaq/train.log

CUDA_VISIBLE_DEVICES=0 python -u train_liqe_single.py > exp_log/leave_one_out/para/train.log

CUDA_VISIBLE_DEVICES=0 python -u train_liqe_single.py > exp_log/leave_one_out/eva/train.log