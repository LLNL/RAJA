/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template 
 *          methods for execution via CUDA kernel launch.
 *
 *          These methods should work on any platform.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cuda_HXX
#define RAJA_forall_cuda_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "execpolicy.hxx"

#include "fault_tolerance.hxx"

//
// Note: run-time is sensitive to # threads per thread block
//
// RDH TODO come up with something better than this.... 
//
#define THREADS_PER_BLOCK 256


namespace RAJA {


//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes
//
//////////////////////////////////////////////////////////////////////
//
#if 0 // RDH
template <typename T>
class ReduceMin<T> 
{
public:

   explicit ReduceMin(T init_val) 
   {
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
      cudaMallocManaged((void **)& m_minval, sizeof(T), cudaMemAttachGlobal) ;
      *m_minval = init_val ;
      cudaDeviceSynchronize();
   } ;

#if 0
   ~ReduceMin()
   {
      cudaDeviceSynchronize();
      cudaFree(m_minval) ;
   } ;
#endif

   __device__ const ReduceMin<T> operator+=(T val) const {
      __shared__ T sd[THREADS_PER_BLOCK];
      sd[threadIdx.x] = val;
      T temp = 0.0;
      __syncthreads();
      for (int i = THREADS_PER_BLOCK / 2; i >= WARP_SIZE; i /= 2) {
         if (threadIdx.x < i) {
            sd[threadIdx.x] += sd[threadIdx.x + i];
         }
         __syncthreads();
      }
      if (threadIdx.x < WARP_SIZE) {
         temp = sd[threadIdx.x];
         for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
            temp += shfl_xor(temp, i);
         }
      }
      // one thread adds to gmem
      if (threadIdx.x == 0) {
         atomicAdd(m_minval, temp);
      }
      return *this ;
   }

   operator double() const {
      cudaDeviceSynchronize() ;
      return *m_minval ;
   }

private:
   T* m_minval ;
} ;
#endif




//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range index sets.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall method template for index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_kernel(LOOP_BODY loop_body, Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(ii);
  }
}


/*!
 ******************************************************************************
 *
 * \brief  Forall execution over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin) / blockSize + 1;
   forall_kernel<<<gridSize, blockSize>>>(loop_body, end - begin);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}


#if 1 // Original code...
/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall_minloc method template for index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
__global__ void forall_minloc_kernel(LOOP_BODY loop_body, 
                                     T* min, Index_type* loc,
                                     Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(ii, min, loc);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  Forall min-loc reduction over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cuda_exec,
                   Index_type begin, Index_type end,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin) / blockSize + 1;
   forall_minloc_kernel<<<gridSize, blockSize>>>(loop_body, 
                                                 min, loc,
                                                 end - begin);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}

#else // Holger's prototype...



#endif


/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall method template for indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_kernel(LOOP_BODY loop_body, 
                              const Index_type *idx, 
                              Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(idx[ii]);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution for indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec,
            const Index_type* __restrict__ idx, const Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = len / blockSize + 1;
   forall_kernel<<<gridSize, blockSize>>>(loop_body, idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief  General CUDA kernal forall_minloc method template for indirection
 *         array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_minloc_kernel(LOOP_BODY loop_body,
                                     double *min, int *loc,
                                     const Index_type *idx, 
                                     Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(idx[ii], min, loc);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  Forall min-loc reduction for indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cuda_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   double *min, int *loc, 
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = len / blockSize + 1;
   forall_minloc_kernel<<<gridSize, blockSize>>>(loop_body,
                                                 min, loc,
                                                 idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   RAJA_FT_END ;
}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
