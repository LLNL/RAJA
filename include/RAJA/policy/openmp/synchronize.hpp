/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for OpenMP synchronization.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_synchronize_openmp_HPP
#define RAJA_synchronize_openmp_HPP

namespace RAJA
{

namespace policy
{

namespace omp
{

/*!
 * \brief Synchronize all OpenMP threads and tasks.
 */
RAJA_INLINE
void synchronize_impl(const omp_synchronize&)
{
#pragma omp barrier
}


} // end of namespace omp
} // namespace policy
} // end of namespace RAJA

#endif // RAJA_synchronize_openmp_HPP
