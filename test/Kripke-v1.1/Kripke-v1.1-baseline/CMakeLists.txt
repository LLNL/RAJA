include_directories(.)

add_subdirectory(Kripke)

add_executable(Kripke-v1.1-baseline.exe "Kripke.cpp")

target_link_libraries(Kripke-v1.1-baseline.exe ${KRIPKE_LIBS} ${KRIPKE_LIBS})

