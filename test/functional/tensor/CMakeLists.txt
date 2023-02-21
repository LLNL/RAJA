###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set( TENSOR_ELEMENT_TYPES  int32_t int64_t float double )

add_subdirectory(register)
#add_subdirectory(vector)
add_subdirectory(matrix)


unset( TENSOR_ELEMENT_TYPES )
