add_definitions(-DRAJA_USE_CYCLE)

file(COPY luleshRAJA-parallel.cxx.src
  DESTINATION ${PROJECT_BINARY_DIR}/test/LULESH-v1.0/LULESH-v1.0_RAJA-CPUvariants)
file(RENAME ${PROJECT_BINARY_DIR}/test/LULESH-v1.0/LULESH-v1.0_RAJA-CPUvariants/luleshRAJA-parallel.cxx.src
  ${PROJECT_BINARY_DIR}/test/LULESH-v1.0/LULESH-v1.0_RAJA-CPUvariants/luleshRAJA-parallel.cxx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(lulesh-RAJA-parallel.exe
  ${PROJECT_BINARY_DIR}/test/LULESH-v1.0/LULESH-v1.0_RAJA-CPUvariants/luleshRAJA-parallel.cxx)

#add_executable(lulesh-RAJA-serial.exe
#  luleshRAJA-serial.cxx)

target_link_libraries(lulesh-RAJA-parallel.exe
  RAJA)

#target_link_libraries(lulesh-RAJA-serial.exe
# RAJA)
