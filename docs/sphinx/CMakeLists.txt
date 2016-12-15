set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py"
        "${SPHINX_BUILD_DIR}/conf.py"
        @ONLY)

add_custom_target(docs
        ${SPHINX_EXECUTABLE}
        -q -b html
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
        COMMENT "Building HTML documentation with Sphinx")

install(DIRECTORY "${SPHINX_HTML_DIR}"
        DESTINATION "docs/sphinx/" OPTIONAL)
