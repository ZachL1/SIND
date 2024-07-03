

# CUDA_VISIBLE_DEVICES=1 python pyiqa/train.py -opt options/train/TOPIQ/train_TOPIQ_res50_koniq_spaq.yml
# CUDA_VISIBLE_DEVICES=1 python pyiqa/train.py -opt options/train/TOPIQ/train_TOPIQ_res50_spaq.yml


# CUDA_VISIBLE_DEVICES=0 python pyiqa/train.py -opt options/train/HyperNet/train_HyperNet_koniq_spaq.yml
# CUDA_VISIBLE_DEVICES=0 python pyiqa/train.py -opt options/train/HyperNet/train_HyperNet_spaq.yml

# CUDA_VISIBLE_DEVICES=0 python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN_spaq.yml
# CUDA_VISIBLE_DEVICES=0 python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN_koniq10k_spaq.yml


CUDA_VISIBLE_DEVICES=1 python pyiqa/train.py -opt options/train/CLIPIQA/train_CLIPIQA_spaq.yml
CUDA_VISIBLE_DEVICES=1 python pyiqa/train.py -opt options/train/CLIPIQA/train_CLIPIQA_koniq10k_spaq.yml