/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA simd policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_vector_HPP
#define policy_vector_HPP

#include "RAJA/policy/PolicyBase.hpp"
#include<RAJA/policy/register.hpp>

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
namespace RAJA
{
namespace policy
{
namespace vector
{

template<typename VEC_TYPE>
struct vector_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {
  using vector_type = VEC_TYPE;
};



}  // end of namespace vector

}  // end of namespace policy

using policy::vector::vector_exec;



template<typename T, size_t UNROLL = 1, typename REGISTER = policy::register_default>
using StreamVector = StreamVectorExt<
    RAJA::Register<REGISTER, T, RAJA::RegisterTraits<REGISTER, T>::s_num_elem>,
    UNROLL>;

template<typename T, size_t NumElem, typename REGISTER = policy::register_default>
using FixedVector = FixedVectorExt<
    RAJA::Register<REGISTER, T, RAJA::RegisterTraits<REGISTER, T>::s_num_elem>,
    NumElem>;


template<typename T, camp::idx_t ROWS, camp::idx_t COLS, typename REGISTER_POLICY  = policy::register_default>
using FixedMatrix =
    internal::MatrixImpl<typename internal::FixedVectorTypeHelper<Register<REGISTER_POLICY, T, 4>, (size_t)COLS>::type, MATRIX_ROW_MAJOR, camp::make_idx_seq_t<internal::FixedVectorTypeHelper<Register<REGISTER_POLICY, T, 4>, COLS>::s_num_registers>, camp::make_idx_seq_t<ROWS>, camp::make_idx_seq_t<COLS> >;
    //internal::MatrixImpl<FixedVector<T, COLS, REGISTER>, MATRIX_ROW_MAJOR, camp::make_idx_seq_t<1>, camp::make_idx_seq_t<ROWS>, camp::make_idx_seq_t<COLS> >;



}  // end of namespace RAJA

#endif
