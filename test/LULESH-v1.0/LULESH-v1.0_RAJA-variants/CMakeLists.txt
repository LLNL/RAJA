include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (RAJA_USE_CUDA)
  cuda_add_executable(lulesh-RAJA-parallel.exe
    luleshRAJA-parallel.cxx)
else()
  add_executable(lulesh-RAJA-parallel.exe
    luleshRAJA-parallel.cxx)
endif()

add_executable(lulesh-RAJA-serial.exe
  luleshRAJA-serial.cxx)

target_link_libraries(lulesh-RAJA-parallel.exe
  RAJA)

target_link_libraries(lulesh-RAJA-serial.exe
 RAJA)
