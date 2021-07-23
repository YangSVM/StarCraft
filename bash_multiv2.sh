#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/codes/marl/smac

for i in {8..9} ;
do
    ( python main.py --alg=qmix --map=2s3z --n_tasks=5 --multi_process_n=$i & )
done
