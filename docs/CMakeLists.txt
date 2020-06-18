###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

add_custom_target(docs)

if (SPHINX_FOUND)
  add_subdirectory(sphinx/user_guide)
endif()

if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif()

if ( NOT SPHINX_FOUND AND NOT DOXGEN_FOUND)
  message(WARNING "ENABLE_DOCUMENTATION=On, but Sphinx or Doxygen not found. \
    Documentation won't be built.")
endif ()
