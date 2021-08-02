#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/codes/marl/smac

for i in {8..9} ;
do
    ( python main.py --alg=qmix --map=2s3z --task_dec_type=n_enemy --multi_process_n=$i & )
done
