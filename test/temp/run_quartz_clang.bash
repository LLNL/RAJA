#!/bin/bash

module add clang/4.0.0

rm *.o clang40.x
clang++-4.0.0 -std=c++14 -O3     -march=native -Wno-vla  -I../../include -I../../install-gcc-4.9.3/include -I. -c arrayAccessorPerformance.cpp
clang++-4.0.0 -std=c++14 -O3     -march=native -Wno-vla  -I../../include -I../../install-gcc-4.9.3/include -I. -c main.cpp
clang++-4.0.0 -std=c++14 -O3     -march=native -o clang40.x  main.o arrayAccessorPerformance.o

sleep 1
for i in `seq 1 10`;
do
  ./clang40.x    $1 $2 $3 100 2 1
    sleep 1
done