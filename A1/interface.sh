#!/bin/bash

ED=$1;
path_in=$2;
path_out=$3;

if [ $ED == C ] 
then
	./encrypt $2 $3
else
	./decrypt $2 $3
fi

