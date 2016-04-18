/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#ifndef RAJA_forallN_cuda_HXX__
#define RAJA_forallN_cuda_HXX__

#include<RAJA/config.hxx>
#include<RAJA/int_datatypes.hxx>

namespace RAJA {


struct CudaDim {
  dim3 num_threads;
  dim3 num_blocks;
  
  __host__ __device__ void print(void) const {
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
  
  CudaThreadBlock(CudaDim &dims, RangeSegment const &is) : 
    begin(is.getBegin()), end(is.getEnd())
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

template<int THREADS>
using cuda_threadblock_x_exec = CudaPolicy<CudaThreadBlock<Dim3x, THREADS>>;

template<int THREADS>
using cuda_threadblock_y_exec = CudaPolicy<CudaThreadBlock<Dim3y, THREADS>>;

template<int THREADS>
using cuda_threadblock_z_exec = CudaPolicy<CudaThreadBlock<Dim3z, THREADS>>;




template<typename VIEWDIM>
struct CudaThread {
  int begin;
  int end;
  
  VIEWDIM view;
  
  CudaThread(CudaDim &dims, RangeSegment const &is) : 
    begin(is.getBegin()), end(is.getEnd())
  {  
    setDims(dims);
  }

  __device__ inline int operator()(void){
    
    int idx = begin + view(threadIdx);
    if(idx >= end){
      idx = -1;
    }
    return idx;
  }
  
  void inline setDims(CudaDim &dims){
    int n = end-begin;
    view(dims.num_threads) = n;  
  }  
};

using cuda_thread_x_exec = CudaPolicy<CudaThread<Dim3x>>;

using cuda_thread_y_exec = CudaPolicy<CudaThread<Dim3y>>;

using cuda_thread_z_exec = CudaPolicy<CudaThread<Dim3z>>;





template<typename VIEWDIM>
struct CudaBlock {
  int begin;
  int end;
  
  VIEWDIM view;
  
  CudaBlock(CudaDim &dims, RangeSegment const &is) : 
    begin(is.getBegin()), end(is.getEnd())
  {  
    setDims(dims);
  }

  __device__ inline int operator()(void){
    
    int idx = begin + view(blockIdx);
    if(idx >= end){
      idx = -1;
    }
    return idx;
  }
  
  void inline setDims(CudaDim &dims){
    int n = end-begin;
    view(dims.num_blocks) = n;  
  }  
};

using cuda_block_x_exec = CudaPolicy<CudaBlock<Dim3x>>;

using cuda_block_y_exec = CudaPolicy<CudaBlock<Dim3y>>;

using cuda_block_z_exec = CudaPolicy<CudaBlock<Dim3z>>;





// Function to check indices for out-of-bounds
template <typename BODY, typename ... ARGS>
RAJA_INLINE
__device__ void cudaCheckBounds(BODY body, int i, ARGS ... args){
  if(i >= 0){
    ForallN_BindFirstArg<BODY, Index_type> bound(body, i);
    cudaCheckBounds(bound, args...);
  }  
}

template <typename BODY>
RAJA_INLINE
__device__ void cudaCheckBounds(BODY body, int i){
  if(i >= 0){
    body(i);
  }  
}

// Launcher that uses execution policies to map blockIdx and threadIdx to map
// to N-argument function
template <typename BODY, typename ... CARGS>
__global__ void cudaLauncherN(BODY body, CARGS ... cargs){
  
  // Compute indices and then pass through the bounds-checking mechanism
  cudaCheckBounds(body, (cargs())... );
}


/*
template<int ...> struct integer_sequence {};

template<int N, int ...S> struct gen_sequence : gen_sequence<N-1, N-1, S...> {};

template<int ...S> struct gen_sequence<0, S...>{ typedef integer_sequence<S...> type; };

template<typename CuARG0, typename ISET0, typename ... CuARGS, typename ... ISETS>
struct ForallN_Executor< 
  ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0>,
  ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS>... > 
{

  ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> iset0;  
  std::tuple<ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS>...> isets;
  
  ForallN_Executor(
    ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const & iset0_, 
    ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS> const & ...isets_) 
    :  iset0(iset0_), isets(isets_...) 
  { }

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
    unpackIndexSets(body, typename gen_sequence<sizeof...(CuARGS)>::type()); 
  }
  
  template<typename BODY, int ... N>
  RAJA_INLINE
  void unpackIndexSets(BODY body, integer_sequence<N...>) const {
  
    CudaDim dims;
    
    callLauncher(dims, body, CuARG0(dims, iset0), CuARGS(dims, std::get<N>(isets))...);
  }
  
  
  template<typename BODY, typename ... CARGS>
  RAJA_INLINE
  void callLauncher(CudaDim const &dims, BODY body, CARGS const &... cargs) const {
    cudaLauncherN<<<dims.num_blocks, dims.num_threads>>>(body, cargs...);
    cudaErrchk(cudaPeekAtLastError());
    cudaErrchk(cudaDeviceSynchronize());
  }
};*/


template<typename CuARG0, typename ISET0>
struct ForallN_Executor< 
  ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> > 
{
  ISET0 iset0;
  
  ForallN_Executor(
    ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const & iset0_) 
    :  iset0(iset0_) 
  { }

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
    CudaDim dims;
    CuARG0 c0(dims, iset0);
    
    cudaLauncherN<<<dims.num_blocks, dims.num_threads>>>(body, c0);
    cudaErrchk(cudaPeekAtLastError());
    cudaErrchk(cudaDeviceSynchronize());
  }
};


template<typename CuARG0, typename ISET0, typename CuARG1, typename ISET1>
struct ForallN_Executor< 
  ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0>,
  ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1> > 
{
  ISET0 iset0;
  ISET1 iset1; 
  
  ForallN_Executor(
    ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const & iset0_, 
    ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1> const & iset1_) 
    :  iset0(iset0_), iset1(iset1_)
  { }

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
    CudaDim dims;
    CuARG0 c0(dims, iset0);
    CuARG1 c1(dims, iset1);
    
    cudaLauncherN<<<dims.num_blocks, dims.num_threads>>>(body, c0, c1);
    cudaErrchk(cudaPeekAtLastError());
    cudaErrchk(cudaDeviceSynchronize());
  }
};


template<typename CuARG0, typename ISET0, typename CuARG1, typename ISET1, typename CuARG2, typename ISET2>
struct ForallN_Executor< 
  ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0>,
  ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1>,
  ForallN_PolicyPair<CudaPolicy<CuARG2>, ISET2> > 
{
  ISET0 iset0;
  ISET1 iset1; 
  ISET2 iset2;
  
  ForallN_Executor(
    ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const & iset0_, 
    ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1> const & iset1_,
    ForallN_PolicyPair<CudaPolicy<CuARG2>, ISET2> const & iset2_) 
    :  iset0(iset0_), iset1(iset1_), iset2(iset2_)
  { }

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
    CudaDim dims;
    CuARG0 c0(dims, iset0);
    CuARG1 c1(dims, iset1);
    CuARG2 c2(dims, iset2);
    
    cudaLauncherN<<<dims.num_blocks, dims.num_threads>>>(body, c0, c1, c2);
    cudaErrchk(cudaPeekAtLastError());
    cudaErrchk(cudaDeviceSynchronize());
  }
};

} // namespace RAJA
  
#endif

