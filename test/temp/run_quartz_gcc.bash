#!/bin/bash

module add gcc/7.1.0

rm *.o gcc71.x
g++  -std=c++14 -O3     -march=native -Wno-vla -I../../include  -I../../install-gcc-4.9.3/include -I. -c arrayAccessorPerformance.cpp
g++  -std=c++14 -O3     -march=native -Wno-vla -I../../include  -I../../install-gcc-4.9.3/include -I. -c main.cpp
g++  -std=c++14 -O3     -march=native -o gcc71.x  main.o arrayAccessorPerformance.o

sleep 1
#echo gcc49
for i in `seq 1 10`;
do
  ./gcc71.x    $1 $2 $3 100 2 1
    sleep 1
done