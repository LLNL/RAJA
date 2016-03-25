include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (RAJA_USE_CUDA)
  add_definitions(-DUSE_CASE=9)
  cuda_add_executable(lulesh-RAJA-parallel.exe
    luleshRAJA-parallel.cxx)
else()
  add_executable(lulesh-RAJA-parallel.exe
    luleshRAJA-parallel.cxx)
endif()

#add_executable(lulesh-RAJA-serial.exe
#  luleshRAJA-serial.cxx)

target_link_libraries(lulesh-RAJA-parallel.exe
  RAJA
  ${RT_LIBRARIES})

#target_link_libraries(lulesh-RAJA-serial.exe
# RAJA
# ${RT_LIBRARIES})
