include_directories(.)

add_subdirectory(Kripke)

set(KRIPKE_LIBS ${KRIPKE_LIBS} RAJA) 

add_executable(Kripke-v1.1-RAJA.exe "Kripke.cpp")
target_link_libraries(Kripke-v1.1-RAJA.exe ${KRIPKE_LIBS} ${KRIPKE_LIBS})

