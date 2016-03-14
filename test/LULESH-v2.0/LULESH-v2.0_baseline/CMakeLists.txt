add_definitions(-DUSE_MPI=0 -DUSE_OMP=1)

add_executable(lulesh2.0NORAJA.exe
  lulesh.cc
  lulesh-comm.cc
  lulesh-init.cc
  lulesh-util.cc
  lulesh-viz.cc)
