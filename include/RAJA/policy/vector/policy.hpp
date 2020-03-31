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
#include "RAJA/config.hpp"
#include "RAJA/pattern/vector.hpp"

//
//////////////////////////////////////////////////////////////////////
//
// SIMD register types and policies
//
//////////////////////////////////////////////////////////////////////
//




#ifdef __AVX2__
#include<RAJA/policy/vector/register/avx2.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::avx2_register
#endif
#endif


#ifdef __AVX__
#include<RAJA/policy/vector/register/avx.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::avx_register
#endif
#endif



#ifdef RAJA_ALTIVEC
#include<RAJA/policy/vector/register/altivec.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::altivec_register
#endif
#endif


// The scalar register is always supported (doesn't require any SIMD/SIMT)
#include<RAJA/policy/vector/register/scalar/scalar.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::scalar_register
#endif


namespace RAJA
{
namespace policy
{
    // This sets the default SIMD register that will be used
    using register_default = RAJA_VECTOR_REGISTER_TYPE;

} // namespace policy
} // namespace RAJA


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



}  // end of namespace RAJA

#endif
