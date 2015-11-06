add_definitions(-DRAJA_USE_CLOCK)
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

cuda_add_executable(reducemin.exe
  ReduceMin.cu)

cuda_add_executable(reducemax.exe
  ReduceMax.cu)

cuda_add_executable(reducesum.exe
  ReduceSum.cu)

cuda_add_executable(traversal.exe
  Traversal.cu)

target_link_libraries(reducemin.exe RAJA)
target_link_libraries(reducemax.exe RAJA)
target_link_libraries(reducesum.exe RAJA)
target_link_libraries(traversal.exe RAJA)
