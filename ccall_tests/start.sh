#!/bin/bash

g++ -c -Wall -Werror -fpic lib_tmp.cpp 
g++ -shared -o libtmp.so lib_tmp.o 
g++ -L/home/agrucelski/Dokumenty/git_mw/ccall_tets/ -Wall -o test main.cpp -ltmp `pkg-config --cflags --libs opencv`
