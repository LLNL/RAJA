add_custom_target(docs)

if (SPHINX_FOUND)
  add_subdirectory(sphinx)
endif()

if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif()

if ( NOT SPHINX_FOUND AND NOT DOXGEN_FOUND)
  message(WARNING "ENABLE_DOCUMENTATION=On, but Sphinx or Doxygen not found. \
    Documentation won't be built.")
endif ()
