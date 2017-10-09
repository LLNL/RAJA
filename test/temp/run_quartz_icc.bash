#!/bin/bash

module add intel/17.0.2

rm *.o icc17.x
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -I.  -I../../include    -I../../install-icpc-1cpc-17.0.174/include  -c arrayAccessorPerformance.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -I.  -I../../include    -I../../install-icpc-1cpc-17.0.174/include -c main.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source     -o icc17.x  main.o arrayAccessorPerformance.o

sleep 1
#echo icc17
for i in `seq 1 10`;
do
  ./icc17.x  $1 $2 $3 100 2 1
  sleep 1
done
