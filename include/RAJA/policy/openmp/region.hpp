//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
    body();
    }
}

}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
