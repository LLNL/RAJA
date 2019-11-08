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

#ifndef policy_simd_HPP
#define policy_simd_HPP

#include "RAJA/policy/PolicyBase.hpp"

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
namespace simd
{

struct simd_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {
};

template<typename VECTOR_TYPE>
struct simd_vector_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {

  using vector_type = VECTOR_TYPE;
};





template<typename IDX, typename VECTOR_TYPE>
class VectorIndex {
  public:
    using index_type = IDX;
    using vector_type = VECTOR_TYPE;

    RAJA_INLINE
    constexpr
    VectorIndex() : m_index(0), m_length(vector_type::s_num_elem) {}

    RAJA_INLINE
    constexpr
    VectorIndex(index_type value, size_t length) : m_index(value), m_length(length) {}

    RAJA_INLINE
    constexpr
    index_type operator*() const {
      return m_index;
    }

    RAJA_INLINE
    constexpr
    size_t size() const {
      return m_length;
    }

  private:
    index_type m_index;
    size_t m_length;
};




struct simd_register{};

}  // end of namespace simd

}  // end of namespace policy

using policy::simd::simd_exec;
using policy::simd::simd_vector_exec;
using policy::simd::VectorIndex;
using policy::simd::simd_register;

}  // end of namespace RAJA

#endif
