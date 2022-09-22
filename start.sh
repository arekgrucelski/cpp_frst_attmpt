#!/bin/bash

##/*
# Author: Arkadiusz Grucelski
# 
# The file belong to AGICortex (to use and modify).
# 
#  Starting script fro testing of ccall (CxxWrap) for naive convolution
# \todo the grid of tests can be started from here
# \todo the output is hard to understand
##*/

cxxwrap_drnm=$(dirname `find ~/.julia/ -name "libcxxwrap_julia.so"`| head -n 1)
tst_drnm=$(pwd)
jlhdr_drnm=$(dirname `locate julia.h |grep include |tail -n 1`)
jllib_drnm=$(dirname `locate -b '\libjulia.so.1' |tail -n 1`)

LD_LIBRARY_PATH=$cxxwrap_drnm
LD_LIBRARY_PATH+=":"$tst_drnm
LD_LIBRARY_PATH+=":"$jllib_drnm
export LD_LIBRARY_PATH
##echo $LD_LIBRARY_PATH
##
g++ -std=c++1z -I$jlhdr_drnm -DCxxJULIA -c -Wall -Werror -fpic src/tst.cpp
g++ -shared -o libtst.so tst.o 
##`pkg-config --cflags --libs julia`
##`pkg-config --cflags --libs opencv`

##`pkg-config --cflags --libs cxxwrap_julia`
g++ -std=c++1z -I$jlhdr_drnm -L$jllib_drnm -L$cxxwrap_drnm -L$tst_drnm -DCxxJULIA -Wall -o test src/main.cpp -ltst -ljulia -lcxxwrap_julia
##`pkg-config --cflags --libs opencv`

#echo $jlhdr_drnm
#echo $jllib_drnm
for x in {4..13}
do
    let WIDTH=120*x # 6
    let HEIGHT=120*x
    echo $WIDTH $HEIGHT
    sed -i "10s/#define W.*/#define WIDTH $WIDTH/" src/tst.h
    sed -i "11s/#define H.*/#define HEIGHT $HEIGHT/" src/tst.h
    #echo "C++ "
    ./test

    #echo "Julia "
    julia src/hello.jl $HEIGHT $WIDTH 
done 
