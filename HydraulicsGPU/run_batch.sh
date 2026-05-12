#!/bin/bash

for ((i=1; i<=9; i++))
do
    val=$((2**(i+9)))
    qsub -N "TestRun_$val" run.sh $val
done