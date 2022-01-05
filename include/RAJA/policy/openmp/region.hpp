//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_region_openmp_HPP
#define RAJA_region_openmp_HPP

namespace RAJA
{
namespace policy
{
namespace omp
{

/*!
 * \brief RAJA::region implementation for OpenMP.
 *
 * Generates an OpenMP parallel region
 *
 * \code
 *
 * RAJA::region<omp_parallel_region>([=](){
 *
 *  // region body - may contain multiple loops
 *
 *  });
 *
 * \endcode
 *
 * \tparam Policy region policy
 *
 */

template <typename Func>
RAJA_INLINE void region_impl(const omp_parallel_region &, Func &&body)
{

#pragma omp parallel
    { // curly brackets to ensure body() is encapsulated in omp parallel region
      //thread private copy of body
      auto loopbody = body;
      loopbody();
    }
}

}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
