include_directories(.)

add_subdirectory(Kripke)

if(RAJA_USE_CUDA)
  cuda_add_executable(Kripke-v1.1-baseline.exe "Kripke.cpp")
else()
  add_executable(Kripke-v1.1-baseline.exe "Kripke.cpp")
endif()

target_link_libraries(Kripke-v1.1-baseline.exe ${KRIPKE_LIBS} ${KRIPKE_LIBS})

