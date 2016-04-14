##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
## 
## Produced at the Lawrence Livermore National Laboratory.
## 
## All rights reserved.
## 
## For release details and restrictions, please see raja/README-license.txt
##

if (RAJA_USE_CUDA)
  cuda_add_executable(CPUtraversal-test.exe
    main-traversal.cxx
    buildIndexSet.cxx)

  cuda_add_executable(CPUreduce-test.exe
    main-reduce.cxx
    buildIndexSet.cxx)
    
  cuda_add_executable(CPUnested-test.exe
    main-nested.cxx)
    
  cuda_add_executable(CPUnested_reduce-test.exe
    main-nested-reduce.cxx)

else()
  add_executable(CPUtraversal-test.exe
    main-traversal.cxx
    buildIndexSet.cxx)

  add_executable(CPUreduce-test.exe
    main-reduce.cxx
    buildIndexSet.cxx)
  
  add_executable(CPUnested-test.exe
    main-nested.cxx)
    
  add_executable(CPUnested_reduce-test.exe
    main-nested-reduce.cxx)

endif()

target_link_libraries(CPUtraversal-test.exe RAJA)
target_link_libraries(CPUreduce-test.exe RAJA)
target_link_libraries(CPUnested-test.exe RAJA)
target_link_libraries(CPUnested_reduce-test.exe RAJA)
