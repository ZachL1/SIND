# loop 0-7
# run the program 8 times
# CUDA_VISIBLE_DEVICES=i python -u train.py > log_$i.txt 2>&1 &
# for i in {0..7}
# do
#     CUDA_VISIBLE_DEVICES=$i python -u train.py $i > log_$i.txt 2>&1 &
# done

# CUDA_VISIBLE_DEVICES=0 python -u train.py 0 > log_8.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 python -u train.py 1 > log_9.txt 2>&1 &



## cross dataset experiment
CUDA_VISIBLE_DEVICES=0 python -u train.py -1 > log_spaq.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -u train.py -2 > log_mix_koniq_spaq.txt 2>&1 &