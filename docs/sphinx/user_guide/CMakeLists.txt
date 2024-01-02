###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py"
        "${SPHINX_BUILD_DIR}/conf.py"
        @ONLY)

add_custom_target(raja-userguide-sphinx
        ${SPHINX_EXECUTABLE}
        -q -b html
        -c "${SPHINX_BUILD_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
        COMMENT "Building HTML user documentation with Sphinx")

install(DIRECTORY "${SPHINX_HTML_DIR}"
        DESTINATION "docs/user_guide/sphinx/" OPTIONAL)

add_dependencies(raja-docs
  raja-userguide-sphinx)
