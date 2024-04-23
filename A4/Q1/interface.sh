#!/bin/bash

if [ "$1" == "train" ]; then
    python train.py $2 $3
elif [ "$1" == "test" ]; then
    python test.py $2 $3 $4
else
    echo "Invalid option"
fi