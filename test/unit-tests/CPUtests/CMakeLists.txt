add_executable(CPUtraversal-test
  main-traversal.cxx
  buildIndexSet.cxx)

add_executable(CPUreduce-test
  main-reduce.cxx
  buildIndexSet.cxx)

target_link_libraries(CPUtraversal-test RAJA)
target_link_libraries(CPUreduce-test RAJA)
