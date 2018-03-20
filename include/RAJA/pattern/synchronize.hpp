//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
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
template <typename Policy>
void synchronize()
{
  synchronize_impl(Policy{});
}
}

#endif  // RAJA_synchronize_HPP
