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

#include "internal/VectorBase.hpp"
#include "internal/stats.hpp"

namespace RAJA
{


  template<typename T, camp::idx_t UNROLL = 1, typename REGISTER_POLICY = policy::register_default>
  using StreamVector =
      internal::makeVectorImpl<VECTOR_STREAM,
                               REGISTER_POLICY,
                               T,
                               UNROLL*RAJA::Register<REGISTER_POLICY, T>::s_num_elem>;

  template<typename T, camp::idx_t NUM_ELEM, typename REGISTER_POLICY = policy::register_default>
  using FixedVector =
      internal::makeVectorImpl<VECTOR_FIXED,
                               REGISTER_POLICY,
                               T,
                               NUM_ELEM>;

  template<typename VECTOR, camp::idx_t NUM_ELEM>
  using changeVectorLength = typename internal::VectorNewLengthHelper<VECTOR, NUM_ELEM>::type;

}  // namespace RAJA


#endif
