#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/codes/marl/smac

for i in {1..2} ;
do
    ( python main.py --alg=task_decomposition --map=5m_vs_6m --n_tasks=6 --multi_process_n=$i & )
done
