###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

raja_add_test(
  NAME test-plugin
  SOURCES test_plugin.cpp plugin_for_test.cpp)
