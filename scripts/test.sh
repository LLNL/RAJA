#!/bin/bash

HOST_CONFIGS=$(ls ../host-configs/linux/*)

OPTIONS='Release RelWithDebInfo Debug'

for conf in $HOST_CONFIGS; do
  for opt in $OPTIONS; do
    ./config-build.py -hc $conf -bt $opt
  done
done

for i in $(ls build-*); do
  cd $i
  make -j 16
  cd ../
done
