/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_vector_vector_HPP
#define RAJA_pattern_vector_vector_HPP

#include "RAJA/pattern/vector/internal/VectorImpl.hpp"

namespace RAJA
{



  template<typename T, camp::idx_t UNROLL = 1, typename REGISTER_POLICY = policy::register_default>
  using StreamVector = typename
      internal::VectorTypeHelper<REGISTER_POLICY,
                                 T,
                                 UNROLL*RAJA::RegisterTraits<REGISTER_POLICY, T>::num_elem()>::stream_type;

  template<typename T, camp::idx_t NUM_ELEM, typename REGISTER_POLICY = policy::register_default>
  using FixedVector = typename
      internal::VectorTypeHelper<REGISTER_POLICY,
                                 T,
                                 NUM_ELEM>::fixed_type;


  template<typename VECTOR_TYPE, camp::idx_t NUM_ELEM>
  using changeVectorLength = typename internal::VectorNewLengthHelepr<VECTOR_TYPE, NUM_ELEM>::type;

}  // namespace RAJA


#endif
