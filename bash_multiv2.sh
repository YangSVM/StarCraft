#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/codes/marl/smac

for i in {1..2} ;
do
    ( python main.py --alg=task_decomposition_all_without_task --map=2s3z --task_dec_type=n_enemy --multi_process_n=$i & )
done
