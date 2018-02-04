/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for ROCM reductions and other operations.
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

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include "RAJA/policy/rocm/MemUtils_ROCm.hpp"


//namespace RAJA
//{

// basic device memory allocation
using namespace RAJA;
rocmError_t rocmHostAlloc(void ** ptr, size_t nbytes, int device)
{
  hc::accelerator acc;  // default device for now
  *ptr = hc::am_alloc(nbytes,acc,2);
  return rocmPeekAtLastError();
}
rocmError_t rocmFreeHost(void * ptr)
{
  hc::am_free(ptr);
  return rocmPeekAtLastError();
}

// basic device memory allocation
void * rocmDeviceAlloc(size_t nbytes, int device)
{
    void* ptr;
    hc::accelerator acc;  // default device for now
    ptr = hc::am_alloc(nbytes,acc,0);
    rocmErrchk(rocmPeekAtLastError());
    return ptr;
}
rocmError_t rocmMalloc(void ** ptr, size_t nbytes, int device)
{
    hc::accelerator acc;  // default device for now
    *ptr = hc::am_alloc(nbytes,acc,0);
    return rocmPeekAtLastError();
}

rocmError_t rocmMallocManaged(void ** ptr, size_t nbytes, int device)
{
// not really UM allocation
// RAJA seems to only use the MemAttachGlobal(=1) flag (memory can be accessed
//   by any stream), but does sometimes use the default which is 0.
// host pinned allocation
// flag = 1, non-coherent, host resident, but with gpu address space pointer
//           visible from all GPUs
// flag = 2, coherent, host resident, but with host address space pointer
    hc::accelerator acc;  // default device for now
    *ptr = hc::am_alloc(nbytes,acc,1);
    return rocmPeekAtLastError();
}

rocmError_t  rocmDeviceFree(void * ptr)
{
  hc::am_free(ptr);
  return rocmPeekAtLastError();
}
rocmError_t  rocmFree(void * ptr)
{
  hc::am_free(ptr);
  return rocmPeekAtLastError();
}

// memset for GPU device memory
//
using namespace hc;

rocmError_t rocmMemset(void * ptr, unsigned char value, size_t nbytes)
{
  unsigned char * cptr = (unsigned char *) ptr;
  uint32_t  * wptr = (uint32_t *) ptr;
  uint32_t fill = (uint32_t)value + (((uint32_t)value)<<8) +
                  (((uint32_t)value)<<16) + (((uint32_t)value)<<24);
  int n = nbytes/4;
  int r = nbytes - n*4;
  if(n+r)
  {
    extent<1> e(n+(r?r:0));
//TODO:  identify device associated with ptr and launch pfe on that device
    parallel_for_each(e,  [=] (index<1> idx) [[hc]]
    {
      if(idx[0] < n) wptr[idx[0]] = fill;
      if(r)
        if(idx[0] < r)
          cptr[n*4+idx[0]] = value;
    }).wait();
  }
  return rocmPeekAtLastError();
}

// host-to-device and device-to-host copy
rocmError_t rocmMemcpy(void * src, void * dst, size_t size)
{
  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();
  av.copy( src , dst , size);
  return rocmPeekAtLastError();
}
namespace RAJA
{
namespace rocm
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
rocmInfo g_status;

//! State of the host code in this thread
rocmInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

//  THis looks too specific to cuda.  probably remove this file.
//! State of raja rocm stream synchronization for rocm reducer objects
std::unordered_map<rocmStream_t, bool> g_stream_info_map{ {rocmStream_t(0), true} };

}  // closing brace for detail namespace

}  // closing brace for rocm namespace

}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_ENABLE_ROCM)
