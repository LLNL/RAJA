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

#ifndef RAJA_synchronize_cuda_HPP
#define RAJA_synchronize_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

namespace RAJA
{

namespace policy
{

namespace cuda
{

/*!
 * \brief Synchronize the current CUDA device.
 */
RAJA_INLINE
void synchronize_impl(const cuda_synchronize&)
{
  cudaErrchk(cudaDeviceSynchronize());
}


}  // end of namespace cuda
}  // end of namespace impl
}  // end of namespace RAJA

#endif  // defined(RAJA_ENABLE_CUDA)

#endif  // RAJA_synchronize_cuda_HPP
