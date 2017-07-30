raja_add_test(
  NAME chai-policy-tests
  SOURCES chai-policy-tests.cxx)

raja_add_test(
  NAME chai-tests
  SOURCES chai-tests.cxx)

if (RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME chai-nested
    SOURCES chai-nested.cxx)
endif()
