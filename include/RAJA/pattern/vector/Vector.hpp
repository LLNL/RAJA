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


  template<typename REGISTER_TYPE, camp::idx_t NUM_ELEM>
  using FixedVectorExt = typename internal::FixedVectorTypeHelper<REGISTER_TYPE, NUM_ELEM>::type;

  template<typename REGISTER_TYPE, camp::idx_t NUM_REGISTERS>
  using StreamVectorExt = internal::VectorImpl<internal::list_of_n<REGISTER_TYPE, NUM_REGISTERS>, camp::make_idx_seq_t<NUM_REGISTERS>, false>;



  template<typename T, camp::idx_t UNROLL = 1, typename REGISTER = policy::register_default>
  using StreamVector = StreamVectorExt<
      RAJA::Register<REGISTER, T, RAJA::RegisterTraits<REGISTER, T>::num_elem()>,
      UNROLL>;

  template<typename T, camp::idx_t NumElem, typename REGISTER = policy::register_default>
  using FixedVector = FixedVectorExt<
      RAJA::Register<REGISTER, T, RAJA::RegisterTraits<REGISTER, T>::num_elem()>,
      NumElem>;



}  // namespace RAJA


#endif
