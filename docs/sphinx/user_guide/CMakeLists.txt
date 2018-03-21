###############################################################################
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
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

add_dependencies(docs
  raja-userguide-sphinx)
