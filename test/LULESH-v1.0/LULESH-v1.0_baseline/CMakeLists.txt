##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
## 
## Produced at the Lawrence Livermore National Laboratory.
## 
## All rights reserved.
## 
## For release details and restrictions, please see raja/README-license.txt
##

add_executable(lulesh-OMP.exe
  luleshOMP.cc)

target_link_libraries(lulesh-OMP.exe
  ${RT_LIBRARIES})

add_executable(lulesh-OMP_NG.exe
  luleshOMP_NG.cc)

target_link_libraries(
  lulesh-OMP_NG.exe
  ${RT_LIBRARIES})

add_executable(lulesh-serial.exe
  lulesh.cc)

target_link_libraries(
  lulesh-serial.exe
  ${RT_LIBRARIES})
