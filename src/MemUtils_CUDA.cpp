/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

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

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"


//
/////////////////////////////////////////////////////////////////////////////
//
// Device properties for quick lookup
//
/////////////////////////////////////////////////////////////////////////////
//

int RAJA::cuda::device_info::s_num_sm = -1;
int RAJA::cuda::device_info::s_max_threads_per_sm = -1;
int RAJA::cuda::device_info::s_max_shmem_per_block = -1;


namespace RAJA
{

namespace cuda
{

namespace detail
{



//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of execution.
//
/////////////////////////////////////////////////////////////////////////////
//

//! State of the host code globally
cudaInfo g_status;

//! State of the host code in this thread
cudaInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

//! State of raja cuda stream synchronization for cuda reducer objects
std::unordered_map<cudaStream_t, bool> g_stream_info_map{ {cudaStream_t(0), true} };



}  // closing brace for detail namespace

}  // closing brace for cuda namespace

}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_ENABLE_CUDA)
