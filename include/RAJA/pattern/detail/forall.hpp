//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
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
