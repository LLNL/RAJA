/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  RAJA header for execution synchronization template.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_synchronize_HPP
#define RAJA_synchronize_HPP

namespace RAJA
{

/*!
 * \brief Synchronize all current RAJA executions for the specified policy.
 *
 * The type of synchronization performed depends on the execution policy. For
 * example, to syncrhonize the current CUDA device, use:
 *
 * \code
 *
 * RAJA::synchronize<RAJA::cuda_synchronize>();
 *
 * \endcode
 *
 * \tparam Policy synchronization policy
 *
 * \see RAJA::policy::omp::synchronize_impl
 * \see RAJA::policy::cuda::synchronize_impl
 */
template<typename Policy>
void synchronize()
{
  synchronize_impl(Policy {});
}
}  // namespace RAJA

#endif  // RAJA_synchronize_HPP
