##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
## 
## Produced at the Lawrence Livermore National Laboratory.
## 
## All rights reserved.
## 
## For release details and restrictions, please see raja/README-license.txt
##

add_definitions(-DUSE_MPI=0 -DUSE_OMP=1)

add_executable(lulesh2.0NORAJA.exe
  lulesh.cc
  lulesh-comm.cc
  lulesh-init.cc
  lulesh-util.cc
  lulesh-viz.cc)
