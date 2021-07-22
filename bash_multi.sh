#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/tiecun/codes/MARL/simulation/smac


for i in {7..9} ;
do
    ( python main.py --alg=qmix --map=3m --multi_process_n=$i & )
done
