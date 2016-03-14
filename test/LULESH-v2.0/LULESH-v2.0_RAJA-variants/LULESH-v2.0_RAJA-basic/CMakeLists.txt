add_definitions(-DUSE_MPI=0 -DUSE_OMP=1)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (RAJA_USE_CUDA)
  cuda_add_executable(lulesh2.0basic.exe
    lulesh.cc
    lulesh-comm.cc
    lulesh-init.cc
    lulesh-util.cc
    lulesh-viz.cc)
else ()
  add_executable(lulesh2.0basic.exe
    lulesh.cc
    lulesh-comm.cc
    lulesh-init.cc
    lulesh-util.cc
    lulesh-viz.cc)
endif()

target_link_libraries(lulesh2.0basic.exe RAJA)
