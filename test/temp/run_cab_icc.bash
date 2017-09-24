#!/bin/bash

. /usr/local/tools/dotkit/init.sh
use ic-17.0.174

rm *.o icc17.x
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source    -gnu-prefix=/usr/apps/gnu/4.9.3/bin/ -Wl,-rpath,/usr/apps/gnu/4.9.3/lib64  -I.     -I../../install-icpc-1cpc-17.0.174/include  -c arrayAccessorPerformance.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source    -gnu-prefix=/usr/apps/gnu/4.9.3/bin/ -Wl,-rpath,/usr/apps/gnu/4.9.3/lib64  -I.     -I../../install-icpc-1cpc-17.0.174/include -c main.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source    -gnu-prefix=/usr/apps/gnu/4.9.3/bin/ -Wl,-rpath,/usr/apps/gnu/4.9.3/lib64       -o icc17.x  main.o arrayAccessorPerformance.o

sleep 1
#echo icc17
for i in `seq 1 10`;
do
    ./icc17.x  $1 $2 $3 100 2 1
done
