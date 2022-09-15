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

export LD_LIBRARY_PATH='/home/agrucelski/.julia/artifacts/4fcd159fccd2f12b8c8c3da884709dc1de7a30ae/lib/:/home/agrucelski/Dokumenty/git_mw/cpp_frst_attmpt/:/opt/julia-1.6.0/lib'

g++ -std=c++1z -I/opt/julia-1.6.0/include/julia/ -DCxxJULIA -c -Wall -Werror -fpic src/tst.cpp
g++ -shared -o libtst.so tst.o 
##`pkg-config --cflags --libs julia`
##`pkg-config --cflags --libs opencv`

g++ -std=c++1z -L/opt/julia-1.6.0/lib/ -L/home/agrucelski/.julia/artifacts/4fcd159fccd2f12b8c8c3da884709dc1de7a30ae/lib/ -L/home/agrucelski/Dokumenty/git_mw/cpp_frst_attmpt/ -I/opt/julia-1.6.0/include/julia/ -DCxxJULIA -Wall -o test src/main.cpp -ltst -ljulia -lcxxwrap_julia
##`pkg-config --cflags --libs cxxwrap_julia`
##`pkg-config --cflags --libs opencv`

echo "C++ "
./test

echo "Julia "
julia src/hello.jl
