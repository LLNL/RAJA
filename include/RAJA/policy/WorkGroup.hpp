/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA wrapper for "multi-policy" and dynamic policy selection
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Policy_WorkGroup_HPP
#define RAJA_Policy_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"

#include "RAJA/util/concepts.hpp"

namespace RAJA
{

namespace policy
{
namespace workgroup
{

struct ordered
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_order> {
};
struct reverse_ordered
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_order> {
};

struct array_of_pointers
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_storage> {
};
struct ragged_array_of_objects
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_storage> {
};
struct constant_stride_array_of_objects
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_storage> {
};

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename STORAGE_POLICY_T >
struct WorkGroupPolicy
    : public RAJA::make_policy_pattern_platform_t<
                       policy_of<EXEC_POLICY_T>::value,
                       Pattern::workgroup,
                       platform_of<EXEC_POLICY_T>::value> {
  static_assert(RAJA::pattern_is<EXEC_POLICY_T, RAJA::Pattern::workgroup_exec>::value,
      "WorkGroupPolicy: EXEC_POLICY_T must be a workgroup exec policy");
  static_assert(RAJA::pattern_is<ORDER_POLICY_T, RAJA::Pattern::workgroup_order>::value,
      "WorkGroupPolicy: ORDER_POLICY_T must be a workgroup order policy");
  static_assert(RAJA::pattern_is<STORAGE_POLICY_T, RAJA::Pattern::workgroup_storage>::value,
      "WorkGroupPolicy: STORAGE_POLICY_T must be a workgroup storage policy");
};

}  // end namespace workgroup
}  // end namespace policy

using policy::workgroup::ordered;
using policy::workgroup::reverse_ordered;

using policy::workgroup::array_of_pointers;
using policy::workgroup::ragged_array_of_objects;
using policy::workgroup::constant_stride_array_of_objects;

using policy::workgroup::WorkGroupPolicy;

}  // end namespace RAJA

#endif
