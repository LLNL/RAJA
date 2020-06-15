/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA sequential policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_sycl_HPP
#define policy_sycl_HPP

#include "RAJA/policy/PolicyBase.hpp"

#include <cstddef>

namespace RAJA
{

using sycl_dim_t = cl::sycl::range<1>;

namespace detail
{
template <bool Async>
struct get_launch {
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template <>
struct get_launch<false> {
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}  // end namespace detail

namespace policy
{
namespace sycl
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////

template <size_t BLOCK_SIZE, bool Async = false>
struct sycl_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::sycl,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::sycl> {
};

}  // namespace sycl
}  // namespace policy

using policy::sycl::sycl_exec;

}  // namespace RAJA

#endif
