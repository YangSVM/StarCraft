#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/codes/marl/smac


for i in {7..9} ;
do
    ( python main.py --alg=qmix --map=3m --task_dec_type=n_enemy --multi_process_n=$i & )
done
