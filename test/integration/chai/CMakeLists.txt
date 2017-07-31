raja_add_test(
  NAME chai-policy-tests
  SOURCES chai-policy-tests.cpp)

raja_add_test(
  NAME chai-tests
  SOURCES chai-tests.cpp)

if (RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME chai-nested
    SOURCES chai-nested.cpp)
endif()
