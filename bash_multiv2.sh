#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/codes/marl/smac


python main.py --alg=task_decomposition_all_without_task --map=3m --task_dec_type=n_enemy --multi_process_n=3 --lr=5e-5 &

python main.py --alg=task_decomposition_all_without_task --map=3m --task_dec_type=n_enemy --multi_process_n=2 --lr=5e-6&

python main.py --alg=task_decomposition_all_without_task --map=3m --task_dec_type=n_enemy --multi_process_n=1 --lr=5e-3&


