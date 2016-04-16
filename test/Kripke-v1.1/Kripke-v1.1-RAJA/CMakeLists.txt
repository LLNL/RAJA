include_directories(.)

add_subdirectory(Kripke)

set(KRIPKE_LIBS ${KRIPKE_LIBS} RAJA) 

add_definitions(-DRAJA_USE_NESTED)

if(RAJA_USE_CUDA)
  cuda_add_executable(Kripke-v1.1-RAJA.exe "Kripke.cpp")
else()
  add_executable(Kripke-v1.1-RAJA.exe "Kripke.cpp")
endif()

target_link_libraries(Kripke-v1.1-RAJA.exe ${KRIPKE_LIBS} ${KRIPKE_LIBS})

