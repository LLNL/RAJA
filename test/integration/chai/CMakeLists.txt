raja_add_test(
  NAME chai-policy-tests
  SOURCES chai-policy-tests.cxx
  DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

raja_add_test(
  NAME chai-tests
  SOURCES chai-tests.cxx
  DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

raja_add_test(
  NAME chai-nested
  SOURCES chai-nested.cxx
  DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
