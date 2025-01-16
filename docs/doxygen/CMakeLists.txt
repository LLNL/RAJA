###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#------------------------------------------------------------------------------
# Doxygen documentation targets
#------------------------------------------------------------------------------

blt_add_doxygen_target( raja_doxygen )

# Required directory for doxygen to generate output to
add_custom_target( build_doxygen_output_dir
                   ALL
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/html
                 )

add_dependencies( raja_doxygen build_doxygen_output_dir )

add_dependencies(raja_docs
  raja_doxygen)
