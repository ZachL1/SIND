# CUDA_VISIBLE_DEVICES=0 python MetaIQA_FineTune_WILDLIVE.py --train_data spaq

# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data koniq10k

CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data koniq10k_spaq


# CUDA_VISIBLE_DEVICES=1 python MetaIQA_FineTune_WILDLIVE.py --train_data spaq --leave_one_out

# CUDA_VISIBLE_DEVICES=0 python MetaIQA_FineTune_WILDLIVE.py --train_data koniq10k --leave_one_out