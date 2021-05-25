###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

raja_add_executable(
  NAME halo-exchange
  SOURCES halo-exchange.cpp
          lists.cpp
          Item.cpp
          Point.cpp
          Schedule.cpp
          GraphSchedule.cpp
          CopyTransaction.cpp)
