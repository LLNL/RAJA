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

template<typename VALUE_TYPE>
struct simd_fixed_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {

  using value_type = VALUE_TYPE;
};

template<typename VALUE_TYPE>
struct simd_stream_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {

  using value_type = VALUE_TYPE;
};



template<typename IDX, typename REGISTER>
class FixedRegisterIndex {
  public:
    using index_type = IDX;
    using register_type = REGISTER;

    RAJA_INLINE
    FixedRegisterIndex() : m_value(0) {}

    RAJA_INLINE
    explicit FixedRegisterIndex(index_type value) : m_value(value) {}

    RAJA_INLINE
    constexpr
    index_type operator*() const {
      return m_value;
    }

  private:
    index_type m_value;
};


template<typename IDX, typename REGISTER>
class StreamRegisterIndex {
  public:
    using index_type = IDX;
    using register_type = REGISTER;

    RAJA_INLINE
    StreamRegisterIndex() : m_value(0), m_length(REGISTER::s_num_elem) {}

    RAJA_INLINE
    explicit StreamRegisterIndex(index_type value, size_t length) : m_value(value), m_length(length) {}

    RAJA_INLINE
    constexpr
    index_type operator*() const {
      return m_value;
    }

    RAJA_INLINE
    constexpr
    size_t size() const {
      return m_length;
    }

  private:
    index_type m_value;
    size_t m_length;
};




struct simd_register{};

}  // end of namespace simd

}  // end of namespace policy

using policy::simd::simd_exec;
using policy::simd::simd_fixed_exec;
using policy::simd::simd_stream_exec;
using policy::simd::FixedRegisterIndex;
using policy::simd::StreamRegisterIndex;
using policy::simd::simd_register;

}  // end of namespace RAJA

#endif
