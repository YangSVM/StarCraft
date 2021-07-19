#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/tiecun/codes/MARL/simulation/smac

for i in {0..2} ;
do
    ( python main.py --alg=task_decomposition --map=2s3z --multi_process_n=$i & )
done
