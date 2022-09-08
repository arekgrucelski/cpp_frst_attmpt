g++ -std=c++1z -I/opt/julia-1.6.0/include/julia/ -c -Wall -Werror -fpic hello.cpp
g++ -shared -o libhello.so hello.o `pkg-config --cflags --libs opencv`
