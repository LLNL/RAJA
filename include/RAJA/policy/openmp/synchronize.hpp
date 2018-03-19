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


}  // end of namespace omp
}  // end of namespace impl
}  // end of namespace RAJA

#endif  // RAJA_synchronize_openmp_HPP
