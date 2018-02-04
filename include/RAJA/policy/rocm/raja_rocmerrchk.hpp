/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing utility methods used in ROCM operations.
 *
 *          These methods work only on platforms that support ROCM.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_raja_rocmerrchk_HPP
#define RAJA_raja_rocmerrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include <iostream>
#include <string>

#include <hc.hpp>
#include <hc_am.hpp>
#define hipHostMallocDefault        0x0
#include <hip/hcc_detail/hip_runtime_api.h>
#include "RAJA/util/defines.hpp"


namespace RAJA
{
typedef hipError_t rocmError_t;
typedef hipStream_t rocmStream_t;

#define threadIdx_x (hc_get_workitem_id(0))
#define threadIdx_y (hc_get_workitem_id(1))
#define threadIdx_z (hc_get_workitem_id(2))

#define blockIdx_x  (hc_get_group_id(0))
#define blockIdx_y  (hc_get_group_id(1))
#define blockIdx_z  (hc_get_group_id(2))

#define blockDim_x  (hc_get_group_size(0))
#define blockDim_y  (hc_get_group_size(1))
#define blockDim_z  (hc_get_group_size(2))

#define gridDim_x   (hc_get_num_groups(0))
#define gridDim_y   (hc_get_num_groups(1))
#define gridDim_z   (hc_get_num_groups(2))


//namespace rocm
//{


///////////////////////////////////////////////////////////////////////

/*
** routines to mimic cudaErrchk used by many programs.
** we leverage the utilites in hip_runtime_api.h
*/

typedef hipError_t rocmError_t;


inline rocmError_t rocmGetLastError()
{
   return(hipGetLastError());
}

inline rocmError_t rocmPeekAtLastError()
{
   return(hipPeekAtLastError());
}

inline const char *rocmGetErrorName(rocmError_t rocmError)
{
   return(hipGetErrorName(rocmError));
}

inline const char *rocmGetErrorString(rocmError_t rocmError)
{
   return(hipGetErrorString(rocmError));
}

inline void rocmErrorCheck()
{
   if(rocmPeekAtLastError() != hipSuccess)
   {
       printf("rocmError: %s\n",rocmGetErrorString(rocmGetLastError()));
   }
}
//} // namespace rocm

///
///////////////////////////////////////////////////////////////////////
///
/// Utility assert method used in ROCM operations to report ROCM
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define rocmErrchk(ans)                            \
  {                                                \
    ::RAJA::rocmAssert((ans), __FILE__, __LINE__); \
  }

inline void rocmAssert(rocmError_t code,
                       const char *file,
                       int line,
                       bool abort = true)
{
  if (code != hipSuccess) {
    fprintf(
        stderr, "rocmAssert: %s %s %d\n", rocmGetErrorString(code), file, line);
    if (abort) RAJA_ABORT_OR_THROW("ROCMassert");
  }
}

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_ROCM)

#endif  // closing endif for header file include guard
