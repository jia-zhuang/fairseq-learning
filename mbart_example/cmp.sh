#!/usr/bin/env bash

sed -n "$1p" datasets/init/val.zh_CN
echo
sed -n "$1p" epoch6.T.txt
echo
sed -n "$1p" epoch6.H.txt
echo 
sed -n "$1p" epoch12.H.txt
echo 
sed -n "$1p" epoch15.H.txt
echo 
sed -n "$1p" models/init_t5_2gpu_debug1/eval/predictions.txt.2632

