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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
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

struct ordered { };
struct reverse_ordered { };

struct array_of_pointers { };
struct ragged_array_of_objects { };
struct constant_stride_array_of_objects { };

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename STORAGE_POLICY_T >
struct WorkGroupPolicy { };

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
