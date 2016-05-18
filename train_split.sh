#!/bin/bash
dim=64
learning_rate=0.0001
dropout=0.05
log_every=5
num_splits=11
num_threads=8
num_epochs=10

for i in `seq 0 $num_epochs`;
do
    for j in `seq 0 $num_splits`;
    do
        if [ $i = 0 ] && [ $j = 0 ]; then
            OMP_NUM_THREADS=$num_threads python main.py --network dmn_smooth --mode train --epochs 1 --dim $dim --learning_rate $learning_rate --dropout $dropout --log_every $log_every --index $j
        else
            state=`cat last_saved_model.txt`
            OMP_NUM_THREADS=$num_threads python main.py --network dmn_smooth --mode train --epochs 1 --dim $dim --learning_rate $learning_rate --dropout $dropout --log_every $log_every --index $j --load_state $state
        fi
    done    
done
