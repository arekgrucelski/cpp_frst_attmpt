#!/bin/bash

##/*
##Author: Arkadiusz Grucelski
##
##The file belong to AGICortex (to use and modify).
##
## Header for shared library in C++ and Julia
##\todo some mess on level of library linkeage on CxxWrap side
##*/

g++ -std=c++1z -I/opt/julia-1.6.0/include/julia/ -DCxxJULIA -c -Wall -Werror -fpic src/tst.cpp
g++ -shared -o libtst.so tst.o 
##`pkg-config --cflags --libs julia`
##`pkg-config --cflags --libs opencv`

g++ -std=c++1z -L/opt/julia-1.6.0/lib/ -L/home/agrucelski/Dokumenty/git_mw/cpp_frst_attmpt/ -I/opt/julia-1.6.0/include/julia/ -DCxxJULIA -Wall -o test src/main.cpp -ltst -ljulia 
##`pkg-config --cflags --libs cxxwrap_julia`
##`pkg-config --cflags --libs opencv`
