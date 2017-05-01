raja_add_executable(
  NAME chai-tests.exe
  SOURCES chai-tests.cxx
  DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

raja_add_executable(
  NAME chai-nested.exe
  SOURCES chai-nested.cxx
  DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
