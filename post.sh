#!/bin/bash

##/*
# Author: Arkadiusz Grucelski
# 
# The file belong to AGICortex (to use and modify).
# 
# start script and simple op for visualisation 
##*/

bash start.sh > cc

cat cc |grep "C++" > ccc
echo "/n/n" >> ccc
cat cc |grep "Cxx" >> ccc
echo "/n/n" >> ccc
cat cc |grep "ccall" >> ccc
echo "/n/n" >> ccc

gnuplot "plot.gp"

mv *.png cc ccc res
