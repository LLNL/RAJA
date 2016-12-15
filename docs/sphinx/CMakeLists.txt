set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py"
        "${SPHINX_BUILD_DIR}/conf.py"
        @ONLY)

add_custom_target(docs
        ${SPHINX_EXECUTABLE}
        -q -b html
        #-W disable warn on error for now, while our sphinx env is still in flux
        # -c "${SPHINX_BUILD_DIR}"
        # -d "${SPHINX_CACHE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
        COMMENT "Building HTML documentation with Sphinx"
        DEPENDS ${deps})

install(DIRECTORY "${SPHINX_HTML_DIR}"
        DESTINATION "docs/sphinx/" OPTIONAL)