/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  Header file for HIP synchronize method.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_synchronize_hip_HPP
#define RAJA_synchronize_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "RAJA/policy/hip/raja_hiperrchk.hpp"

namespace RAJA
{

namespace policy
{

namespace hip
{

/*!
 * \brief Synchronize the current HIP device.
 */
RAJA_INLINE
void synchronize_impl(const hip_synchronize&)
{
  RAJA_INTERNAL_HIP_CHECK_API_CALL(hipDeviceSynchronize);
}


}  // end of namespace hip
}  // namespace policy
}  // end of namespace RAJA

#endif  // defined(RAJA_ENABLE_HIP)

#endif  // RAJA_synchronize_hip_HPP
