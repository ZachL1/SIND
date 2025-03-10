
# CUDA_VISIBLE_DEVICES=0 python MetaIQA_FineTune_WILDLIVE.py --train_data eva --alignment --leave_one_out &

# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data eva --leave_one_out

# sleep 10m

# CUDA_VISIBLE_DEVICES=0 python MetaIQA_FineTune_WILDLIVE.py --train_data spaq --alignment --leave_one_out &

# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data spaq --leave_one_out --alignment


# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data spaq --alignment

# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data koniq10k

# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data koniq10k_spaq


# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data spaq --leave_one_out

# CUDA_VISIBLE_DEVICES=0 python MetaIQA_FineTune_WILDLIVE.py --train_data koniq10k --leave_one_out