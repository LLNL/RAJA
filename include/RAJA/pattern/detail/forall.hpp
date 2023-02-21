/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  Internal header for RAJA forall "BED" macros
 *         (i.e., loop bounds 'BED' --> begin, end, distance)
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_DETAIL_FORALL_HPP
#define RAJA_PATTERN_DETAIL_FORALL_HPP

#define RAJA_EXTRACT_BED_SUFFIXED(CONTAINER, SUFFIX) \
  using std::begin;                                  \
  using std::end;                                    \
  using std::distance;                               \
  auto begin##SUFFIX = begin(CONTAINER);             \
  auto end##SUFFIX = end(CONTAINER);                 \
  auto distance##SUFFIX = distance(begin##SUFFIX, end##SUFFIX)

#define RAJA_EXTRACT_BED_IT(CONTAINER) RAJA_EXTRACT_BED_SUFFIXED(CONTAINER, _it)

#endif /* RAJA_PATTERN_DETAIL_FORALL_HPP */
