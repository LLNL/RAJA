add_definitions(-DUSE_MPI=0 -DUSE_OMP=1 -DRAJA_USE_GETTIME)

add_executable(lulesh2.0NORAJA
  lulesh.cc
  lulesh-comm.cc
  lulesh-init.cc
  lulesh-util.cc
  lulesh-viz.cc)
