/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_cuda_HXX
#define RAJA_scan_cuda_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_ENABLE_CUDA)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iterator>
#include <type_traits>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>

namespace RAJA
{
namespace detail
{
namespace scan
{

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <typename InputIter, typename Function>
void inclusive_inplace(const ::RAJA::cuda_exec_base&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op)
{
  ::thrust::inclusive_scan(::thrust::device, begin, end, begin, binary_op);
  cudaDeviceSynchronize();
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename InputIter, typename Function, typename T>
void exclusive_inplace(const ::RAJA::cuda_exec_base&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op,
                       T init)
{
  ::thrust::exclusive_scan(
      ::thrust::device, begin, end, begin, init, binary_op);
  cudaDeviceSynchronize();
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename InputIter,
          typename OutputIter,
          typename Function>
void inclusive(const ::RAJA::cuda_exec_base&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op)
{
  ::thrust::inclusive_scan(::thrust::device, begin, end, out, binary_op);
  cudaDeviceSynchronize();
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename InputIter,
          typename OutputIter,
          typename Function,
          typename T>
void exclusive(const ::RAJA::cuda_exec_base&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op,
               T init)
{
  ::thrust::exclusive_scan(::thrust::device, begin, end, out, init, binary_op);
  cudaDeviceSynchronize();
}

}  // closing brace for scan namespace

}  // closing brace for detail namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
