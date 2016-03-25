add_definitions(-DRAJA_USE_CLOCK)
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

cuda_add_executable(reducemin.exe
  ReduceMin.cxx)

cuda_add_executable(reducemax.exe
  ReduceMax.cxx)

cuda_add_executable(reducesum.exe
  ReduceSum.cxx)

cuda_add_executable(traversal.exe
  Traversal.cxx)

target_link_libraries(reducemin.exe RAJA)
target_link_libraries(reducemax.exe RAJA)
target_link_libraries(reducesum.exe RAJA)
target_link_libraries(traversal.exe RAJA)
