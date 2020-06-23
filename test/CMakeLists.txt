###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(include)

raja_add_test_subdirectory(integration)

raja_add_test_subdirectory(functional)

raja_add_test_subdirectory(unit)

raja_add_test_subdirectory(old-tests)

