add_definitions(-DUSE_MPI=0 -DUSE_OMP=1 -DRAJA_USE_CLOCK -DLULESH_HEADER=2)

file(COPY lulesh.cc.src
  DESTINATION ${PROJECT_BINARY_DIR}/test/LULESH-v2.0/LULESH-v2.0_RAJA-MICfriendly)

file(RENAME ${PROJECT_BINARY_DIR}/test/LULESH-v2.0/LULESH-v2.0_RAJA-MICfriendly/lulesh.cc.src
  ${PROJECT_BINARY_DIR}/test/LULESH-v2.0/LULESH-v2.0_RAJA-MICfriendly/lulesh.cc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lulesh2.0_RAJA-ISetMIC.exe
  ${PROJECT_BINARY_DIR}/test/LULESH-v2.0/LULESH-v2.0_RAJA-MICfriendly/lulesh.cc
  lulesh-comm.cc
  lulesh-init.cc
  lulesh-util.cc
  lulesh-viz.cc)

target_link_libraries(lulesh2.0_RAJA-ISetMIC.exe RAJA)
