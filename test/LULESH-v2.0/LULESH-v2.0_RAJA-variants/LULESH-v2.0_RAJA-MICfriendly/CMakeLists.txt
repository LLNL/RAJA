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
add_definitions(-DLULESH_HEADER=2)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (RAJA_USE_CUDA)
  cuda_add_executable(lulesh2.0_RAJA-ISetMIC.exe
    lulesh.cc
    lulesh-comm.cc
    lulesh-init.cc
    lulesh-util.cc
    lulesh-viz.cc)
else()
  add_executable(lulesh2.0_RAJA-ISetMIC.exe
    lulesh.cc
    lulesh-comm.cc
    lulesh-init.cc
    lulesh-util.cc
    lulesh-viz.cc)
endif()

target_link_libraries(lulesh2.0_RAJA-ISetMIC.exe RAJA)
