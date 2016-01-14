add_definitions(-DRAJA_USE_CYCLE)

add_executable(lulesh-OMP.exe
  luleshOMP.cc)

add_executable(lulesh-OMP_NG.exe
  luleshOMP_NG.cc)

add_executable(lulesh-serial.exe
  lulesh.cc)
