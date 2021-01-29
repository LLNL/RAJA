###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (ENABLE_DOCUMENTATION)
  if (SPHINX_FOUND)
    blt_add_sphinx_target( raja_docs )
  endif()

  if (DOXYGEN_FOUND)
    add_subdirectory(doxygen)
  endif()

  if ( NOT SPHINX_FOUND AND NOT DOXGEN_FOUND)
    message(WARNING "ENABLE_DOCUMENTATION=On, but Sphinx or Doxygen not found. \
      Documentation won't be built.")
  endif ()
endif()
