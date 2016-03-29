/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for NVCC CUDA execution.
 *
 *          These methods work only on platforms that support CUDA. 
 *
 ******************************************************************************
 */

#ifndef RAJA_cuda_HXX
#define RAJA_cuda_HXX

#if defined(RAJA_USE_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>


namespace RAJA {


//
/////////////////////////////////////////////////////////////////////
//
// Generalization of CUDA dim3 x, y and z used to describe
// sizes and indices for threads and blocks.
//
/////////////////////////////////////////////////////////////////////
//
#if 0
struct Dim3x {
  __host__ __device__ inline unsigned int &operator()(dim3 &dim){
    return dim.x;
  }

  __host__ __device__ inline unsigned int operator()(dim3 const &dim){
    return dim.x;
  }
};


struct Dim3y {
  __host__ __device__ inline unsigned int &operator()(dim3 &dim){
    return dim.y;
  }

  __host__ __device__ inline unsigned int operator()(dim3 const &dim){
    return dim.y;
  }
};

struct Dim3z {
  __host__ __device__ inline unsigned int &operator()(dim3 &dim){
    return dim.z;
  }

  __host__ __device__ inline unsigned int operator()(dim3 const &dim){
    return dim.z;
  }
};
#endif

//
/////////////////////////////////////////////////////////////////////
//
// Execution policies
//
/////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
template <size_t BLOCK_SIZE>
struct cuda_exec {};
///
template <size_t BLOCK_SIZE>
struct cuda_exec_async {};


//
//
//

//
// NOTE: There is no Index set segment iteration policy for CUDA
//

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
template <size_t BLOCK_SIZE>
struct cuda_reduce {};

//
// Operations in the included files are parametrized using the following
// values. 
//
const int WARP_SIZE = 32;


}  // closing brace for RAJA namespace


//
// Headers containing traversal and reduction templates 
//
#include "reduce_cuda.hxx"
#include "forall_cuda.hxx"


#endif  // closing endif for if defined(RAJA_USE_CUDA)

#endif  // closing endif for header file include guard

