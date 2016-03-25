if (RAJA_USE_CUDA)
  cuda_add_executable(CPUtraversal-test.exe
    main-traversal.cxx
    buildIndexSet.cxx)

  cuda_add_executable(CPUreduce-test.exe
    main-reduce.cxx
    buildIndexSet.cxx)
else()
  add_executable(CPUtraversal-test.exe
    main-traversal.cxx
    buildIndexSet.cxx)

  add_executable(CPUreduce-test.exe
    main-reduce.cxx
    buildIndexSet.cxx)
endif()

target_link_libraries(CPUtraversal-test.exe RAJA)
target_link_libraries(CPUreduce-test.exe RAJA)
