/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#ifndef RAJA_forallN_cuda_HXX__
#define RAJA_forallN_cuda_HXX__

#include<RAJA/config.hxx>
#include<RAJA/int_datatypes.hxx>

namespace RAJA {


struct CudaDim {
  dim3 num_threads;
  dim3 num_blocks;
  
  __host__ __device__ void print(void){
    printf("<<< (%d,%d,%d), (%d,%d,%d) >>>\n",
      num_blocks.x, num_blocks.y, num_blocks.z,
      num_threads.x, num_threads.y, num_threads.z);
  }
};


template<typename POL>
struct CudaPolicy {};

template<typename VIEWDIM, int threads_per_block>
struct CudaThreadBlock {
  int begin;
  int end;
  
  VIEWDIM view;
  
  CudaThreadBlock(CudaDim &dims, int begin0, int end0) : 
    begin(begin0), end(end0)
  {  
    setDims(dims);
  }

  __device__ inline int operator()(void){
    
    int idx = begin + view(blockIdx) * threads_per_block + view(threadIdx);
    if(idx >= end){
      idx = -1;
    }
    return idx;
  }
  
  void inline setDims(CudaDim &dims){
    int n = end-begin;
    if(n < threads_per_block){
      view(dims.num_threads) = n;
      view(dims.num_blocks) = 1;
    }
    else{
      view(dims.num_threads) = threads_per_block;
      
      int blocks = n / threads_per_block;
      if(n % threads_per_block){
        ++ blocks;
      }
      view(dims.num_blocks) = blocks;
    }
  }  
};

template<typename VIEW, int THREADS>
using cuda_threadblock_exec = CudaPolicy<CudaThreadBlock<VIEW, THREADS>>;


// Simple launcher, that maps thread (x,y) to indices
template <typename BODY, typename ... CARGS>
__global__ void cudaLauncherN(BODY body, CARGS ... cargs){
  //int i = ci();
  //int j = cj();
  //if(i >= 0 && j >= 0){
    body((cargs())...);
  //}
}

template<typename CuPI>
struct ForallN_Executor< ForallN_PolicyPair<CudaPolicy<CuPI>, RangeSegment> > {

  RangeSegment const is_i;
  
  ForallN_Executor(RangeSegment const &is_i0) : is_i(is_i0) { }

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
  
    CudaDim dims;
    
    callLauncher(dims, body, CuPI(dims, is_i.getBegin(), is_i.getEnd()));
  }
  
  
  template<typename BODY, typename ... CARGS>
  RAJA_INLINE
  void callLauncher(CudaDim const &dims, BODY body, CARGS const &... cargs) const {
    cudaLauncherN<<<dims.num_blocks, dims.num_threads>>>(body, cargs...);
    cudaErrchk(cudaPeekAtLastError());
    cudaErrchk(cudaDeviceSynchronize());
  }
};




} // namespace RAJA
  
#endif

