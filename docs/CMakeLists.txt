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
