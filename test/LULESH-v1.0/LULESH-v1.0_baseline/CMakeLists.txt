add_executable(lulesh-OMP.exe
  luleshOMP.cc)

target_link_libraries(
  lulesh-OMP.exe
  -lrt)

add_executable(lulesh-OMP_NG.exe
  luleshOMP_NG.cc)

target_link_libraries(
  lulesh-OMP_NG.exe
  -lrt)

add_executable(lulesh-serial.exe
  lulesh.cc)

target_link_libraries(
  lulesh-serial.exe
  -lrt)
