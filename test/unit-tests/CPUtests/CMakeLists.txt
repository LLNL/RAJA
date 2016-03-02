if (RAJA_USE_CUDA)
  cuda_add_executable(CPUtraversal-test
    main-traversal.cxx
    buildIndexSet.cxx)

  cuda_add_executable(CPUreduce-test
    main-reduce.cxx
    buildIndexSet.cxx)
else()
  add_executable(CPUtraversal-test
    main-traversal.cxx
    buildIndexSet.cxx)

  add_executable(CPUreduce-test
    main-reduce.cxx
    buildIndexSet.cxx)
endif()

target_link_libraries(CPUtraversal-test RAJA)
target_link_libraries(CPUreduce-test RAJA)
