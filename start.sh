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

LD_LIBRARY_PATH=$cxxwrap_drnm
LD_LIBRARY_PATH+=":"$tst_drnm
export LD_LIBRARY_PATH+=':/opt/julia-1.6.0/lib'
##
g++ -std=c++1z -I/opt/julia-1.6.0/include/julia/ -DCxxJULIA -c -Wall -Werror -fpic src/tst.cpp
g++ -shared -o libtst.so tst.o 
##`pkg-config --cflags --libs julia`
##`pkg-config --cflags --libs opencv`

g++ -std=c++1z -L/opt/julia-1.6.0/lib/ -L$cxxwrap_drnm -L$tst_drnm -I/opt/julia-1.6.0/include/julia/ -DCxxJULIA -Wall -o test src/main.cpp -ltst -ljulia -lcxxwrap_julia
##`pkg-config --cflags --libs cxxwrap_julia`
##`pkg-config --cflags --libs opencv`

echo "C++ "
./test

echo "Julia "
julia src/hello.jl
