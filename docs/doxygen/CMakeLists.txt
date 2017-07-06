set(DOXYGEN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
set(DOXYGEN_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
  ${DOXYGEN_BUILD_DIR}/Doxyfile 
  @ONLY)

add_custom_target(raja-doxygen
  ${DOXYGEN_EXECUTABLE} 
  ${DOXYGEN_BUILD_DIR}/Doxyfile
  COMMENT "Building API documentation with Doxygen")

install(DIRECTORY ${DOXYGEN_HTML_DIR}
  DESTINATION "docs/doxygen/" OPTIONAL)

add_dependencies(docs
  raja-doxygen)
