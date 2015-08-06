/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template 
 *          methods for execution via CUDA kernel launch.
 *
 *          These methods should work on any platform that supports 
 *          CUDA devices.
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

#if defined(RAJA_USE_CUDA)

#include "execpolicy.hxx"
#include "reducers.hxx"

#include "fault_tolerance.hxx"

#include "MemUtilsCuda.hxx"

#include <iostream>
#include <cstdlib>

namespace RAJA {

//
// Operations in this file are parametrized using the following
// values.  RDH -- can we come up with something better than this??
//
const int THREADS_PER_BLOCK = 256;
const int WARP_SIZE = 32;
const int BLOCK_LENGTH = (1024 + 8) * 16;
const int RAJA_MAX_REDUCE_VARS_CUDA = 8;

__device__ __managed__ int cuda_reduction_gid = -1;

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes and operations.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Min reduction class template for use in CUDA kernel.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<cuda_reduce, T> {
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMin(T init_val) {
       m_init_val = init_val;
       m_blockdata = static_cast<RAJA::CudaReduceBlockAllocType*>(
          allocCudaReductionMemBlockData(BLOCK_LENGTH, RAJA_MAX_REDUCE_VARS_CUDA) ;
       m_myID = ++cuda_reduction_gid;
       m_blockoffset = m_myID * BLOCK_LENGTH;

       if ( m_myID >= RAJA_MAX_REDUCE_VARS_CUDA ) {
          std::cerr << "\n ERROR in CUDA ReduceMin ctor, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
          exit(1);
       }

       m_is_copy = false;

       m_blockdata[m_blockoffset] = init_val;
       cudaDeviceSynchronize();
   }

   //
   // Copy ctor tracks whether object is a copy to work properly.
   //
   __host__ __device__ ReduceMin( const ReduceMin<cuda_reduce, T>& other )
   {
      copy(other);
      m_is_copy = true; 
   }

   //
   // Destructor decrements cuda_reduction_gid value only if reducer object 
   // not created via copy construction (to make copy ctor work properly).
   //
   __host__ __device__ ~ReduceMin() {
      if (!m_is_copy) {
          // call cudaFree on cudaMalloc data if needed.
          cuda_reduction_gid--;
      }
   }

   //
   // Operator to retrieve min value (before object is destroyed).
   //
   operator T() const {
      cudaDeviceSynchronize() ;
      return static_cast<T>(m_blockdata[m_blockoffset]) ;
   }

   //
   // Min function that sets appropriate thread block value to
   // minimum of initial value and current block min.
   //
   __device__ const ReduceMin<cuda_reduce, T> min(T val) const {
      __shared__ T sd[THREADS_PER_BLOCK];

      if(threadIdx.x == 0) {
          for(int i=0;i<THREADS_PER_BLOCK;i++) sd[i] = m_init_val;
      }
      __syncthreads();

      sd[threadIdx.x] = val;
      __syncthreads();

      for (int i = THREADS_PER_BLOCK / 2; i >= WARP_SIZE; i /= 2) {
         if (threadIdx.x < i) {
            sd[threadIdx.x] = RAJA_MIN(sd[threadIdx.x],sd[threadIdx.x + i]);
         }
         __syncthreads();
      }

      if (threadIdx.x < 16) {
          sd[threadIdx.x] = RAJA_MIN(sd[threadIdx.x],sd[threadIdx.x+16]);
      }
      __syncthreads();

      if (threadIdx.x < 8) {
          sd[threadIdx.x] = RAJA_MIN(sd[threadIdx.x],sd[threadIdx.x+8]);
      }
      __syncthreads();

      if (threadIdx.x < 4) {
          sd[threadIdx.x] = RAJA_MIN(sd[threadIdx.x],sd[threadIdx.x+4]);
      }
      __syncthreads();

      if (threadIdx.x < 2) {
          sd[threadIdx.x] = RAJA_MIN(sd[threadIdx.x],sd[threadIdx.x+2]);
      }
      __syncthreads();

      if (threadIdx.x < 1) {
          sd[threadIdx.x] = RAJA_MIN(sd[threadIdx.x],sd[threadIdx.x+1]);
          m_blockdata[m_blockoffset + blockIdx.x+1]  = sd[threadIdx.x];
      }

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMin<cuda_reduce, T>();

   //
   // Copy function for copy-and-swap idiom (shallow).
   //
   void copy(const ReduceMin<cuda_reduce, T>& other)
   {
      m_init_val = other.m_init_val;
      m_blockdata = other.m_blockdata;
      m_blockoffset = other.m_blockoffset;
      m_myID = other.m_myID;
      m_is_copy = true;
   }
   

   T m_init_val;
   CudaReduceBlockAllocType* m_blockdata;
   int m_blockoffset;
   int m_myID;
   bool m_is_copy;
} ;


/*!
 ******************************************************************************
 *
 * \brief  Max reduction class template for use in CUDA kernel.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<cuda_reduce, T> {
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMax(T init_val) {
       m_init_val = init_val;
       m_blockdata = static_cast<RAJA::CudaReduceBlockAllocType*>(
          allocCudaReductionMemBlockData(BLOCK_LENGTH, RAJA_MAX_REDUCE_VARS_CUDA) ;
       m_blockoffset = m_myID * BLOCK_LENGTH;
       m_myID = ++cuda_reduction_gid;

       if ( m_myID >= RAJA_MAX_REDUCE_VARS_CUDA ) {
          std::cerr << "\n ERROR in CUDA ReduceMax ctor, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
          exit(1);
       }

       m_is_copy = false;

       m_blockdata[m_blockoffset] = init_val;
       cudaDeviceSynchronize();
   }

   //
   // Copy ctor tracks whether object is a copy to work properly.
   //
   __host__ __device__ ReduceMax( const ReduceMax<cuda_reduce, T>& other )
   {
      copy(other);
      m_is_copy = true; 
   }

   //
   // Destructor decrements cuda_reduction_gid value only if reducer object 
   // not created via copy construction (to make copy ctor work properly).
   //
   __host__ __device__ ~ReduceMax() {
      if (!m_is_copy) {
          // call cudaFree on cudaMalloc data if needed.
          cuda_reduction_gid--;
      }
   }

   //
   // Operator to retrieve max value (before object is destroyed).
   //
   operator T() const {
       cudaDeviceSynchronize() ;
       return static_cast<T>(m_blockdata[m_blockoffset]) ;
   }

   //
   // Max function that sets appropriate thread block value to
   // maximum of initial value and current block max.
   //
   __device__ const ReduceMax<cuda_reduce, T> max(T val) const {
       __shared__ T sd[THREADS_PER_BLOCK];

       if(threadIdx.x == 0) {
           for(int i=0;i<THREADS_PER_BLOCK;i++) sd[i] = m_init_val;
       }
       __syncthreads();

       sd[threadIdx.x] = val;
       __syncthreads();

       for (int i = THREADS_PER_BLOCK / 2; i >= WARP_SIZE; i /= 2) {
          if (threadIdx.x < i) {
             sd[threadIdx.x] = RAJA_MAX(sd[threadIdx.x],sd[threadIdx.x + i]);
          }
          __syncthreads();
       }

       if (threadIdx.x < 16) {
           sd[threadIdx.x] = RAJA_MAX(sd[threadIdx.x],sd[threadIdx.x+16]);
       }
       __syncthreads();

       if (threadIdx.x < 8) {
           sd[threadIdx.x] = RAJA_MAX(sd[threadIdx.x],sd[threadIdx.x+8]);
       }
       __syncthreads();

       if (threadIdx.x < 4) {
           sd[threadIdx.x] = RAJA_MAX(sd[threadIdx.x],sd[threadIdx.x+4]);
       }
       __syncthreads();

       if (threadIdx.x < 2) {
           sd[threadIdx.x] = RAJA_MAX(sd[threadIdx.x],sd[threadIdx.x+2]);
       }
       __syncthreads();

       if (threadIdx.x < 1) {
           sd[threadIdx.x] = RAJA_MAX(sd[threadIdx.x],sd[threadIdx.x+1]);
           m_blockdata[m_blockoffset + blockIdx.x+1]  = sd[threadIdx.x];
       }

       return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMax<cuda_reduce, T>();

   //
   // Copy function for copy-and-swap idiom (shallow).
   //
   void copy(const ReduceMax<cuda_reduce, T>& other)
   {
      m_init_val = other.m_init_val;
      m_blockdata = other.m_blockdata;
      m_blockoffset = other.m_blockoffset;
      m_myID = other.m_myID;
      m_is_copy = true;
   }
   

   T m_init_val;
   CudaReduceBlockAllocType* m_blockdata;
   int m_blockoffset;
   int m_myID;
   bool m_is_copy;
} ;


/*!
 ******************************************************************************
 *
 * \brief  Sum reduction class template for use in CUDA kernel.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<cuda_reduce, T> {
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceSum(T init_val) {
       m_init_val = init_val;
       m_blockdata = static_cast<RAJA::CudaReduceBlockAllocType*>(
          allocCudaReductionMemBlockData(BLOCK_LENGTH, RAJA_MAX_REDUCE_VARS_CUDA) ;
       m_blockoffset = m_myID * BLOCK_LENGTH;
       m_myID = ++cuda_reduction_gid;

       if ( m_myID >= RAJA_MAX_REDUCE_VARS_CUDA ) {
          std::cerr << "\n ERROR in CUDA ReduceSum ctor, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
          exit(1);
       }

       m_is_copy = false;

       cudaDeviceSynchronize();
   }

   //
   // Copy ctor tracks whether object is a copy to work properly.
   //
   __host__ __device__ ReduceSum( const ReduceSum<cuda_reduce, T>& other )
   {
      copy(other);
      m_is_copy = true; 
   }

   //
   // Destructor decrements cuda_reduction_gid value only if reducer object 
   // not created via copy construction (to make copy ctor work properly).
   //
   __host__ __device__ ~ReduceSum() {
      if (!m_is_copy) {
          // call cudaFree on cudaMalloc data if needed.
          cuda_reduction_gid--;
      }
   }

   //
   // Operator to retrieve sum value (before object is destroyed).
   //
   operator T() const {
       cudaDeviceSynchronize() ;
       return static_cast<T>(m_blockdata[m_blockoffset]);
   }

   //
   // += operator that performs accumulation into the appropriate 
   // thread block value 
   //
   __device__ const ReduceSum<cuda_reduce, T> operator+=(T val) const {
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

      // one thread adds to global memory, we skip m_blockdata[m_blockoffset] 
      // as we're going to accumlate into this
      if (threadIdx.x == 0) {
          m_blockdata[m_blockoffset + blockIdx.x+1] += temp ;
      }
      
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceSum<cuda_reduce, T>();

   //
   // Copy function for copy-and-swap idiom (shallow).
   //
   void copy(const ReduceSum<cuda_reduce, T>& other)
   {
      m_init_val = other.m_init_val;
      m_blockdata = other.m_blockdata;
      m_blockoffset = other.m_blockoffset;
      m_myID = other.m_myID;
      m_is_copy = true;
   }

   //
   // Shuffle operation that I don't understand...
   //
   __device__ __forceinline__
   T shfl_xor(T var, int laneMask)
   {
      int lo = __shfl_xor( __double2loint(var), laneMask );
      int hi = __shfl_xor( __double2hiint(var), laneMask );
      return __hiloint2double( hi, lo );
   }
   

   T m_init_val;
   CudaReduceBlockAllocType* m_blockdata;
   int m_blockoffset;
   int m_myID;
   bool m_is_copy;
} ;


//
//////////////////////////////////////////////////////////////////////
//
// CUDA kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief CUDA kernal forall template for index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_cuda_kernel(LOOP_BODY loop_body,
                                   Index_type begin, Index_type end)
{
   Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
   if (ii < end) {
      loop_body(begin+ii);
   }
}

/*!
 ******************************************************************************
 *
 * \brief CUDA kernal forall_Icount template for index range.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_Icount_cuda_kernel(LOOP_BODY loop_body,
                                          Index_type begin, Index_type end,
                                          Index_type icount)
{
   Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
   if (ii < length) {
      loop_body(ii+icount, ii+begin);
   }
}

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall template for indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_cuda_kernel(LOOP_BODY loop_body, 
                                   const Index_type* idx, 
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
 * \brief  CUDA kernal forall_Icount template for indiraction array.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_Icount_cuda_kernel(LOOP_BODY loop_body,
                                          const Index_type* idx,
                                          Index_type length
                                          Index_type icount)
{
   Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
   if (ii < length) {
      loop_body(ii+icount, idx[ii]);
   }
}


//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over index ranges.
//
////////////////////////////////////////////////////////////////////////
//

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
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                               begin, end);
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
 * \brief  Forall execution over index range with index count,
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cuda_exec,
                   Index_type begin, Index_type end,
                   Index_type icount,
                   LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_Icount_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                                      begin, end,
                                                      icount);
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
 * \brief  Forall min reduction over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(cuda_exec,
                Index_type begin, Index_type end,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               begin, end);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata = 
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
          blockdata[BLOCK_LENGTH*k] =
             RAJA_MIN(blockdata[BLOCK_LENGTH*k],
                        blockdata[(BLOCK_LENGTH*k)+i]) ;
       }
    }

   RAJA_FT_END ;
}


//
// RDH -- need minloc reduction
//


/*!
 ******************************************************************************
 *
 * \brief  Forall max reduction over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_max(cuda_exec,
                Index_type begin, Index_type end,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               begin, end);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata = 
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
          blockdata[BLOCK_LENGTH*k] =
             RAJA_MAX(blockdata[BLOCK_LENGTH*k],
                        blockdata[(BLOCK_LENGTH*k)+i]) ;
       }
    }

   RAJA_FT_END ;
}


//
// RDH -- need maxloc reduction
//


/*!
 ******************************************************************************
 *
 * \brief  Forall sum reduction over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cuda_exec,
                Index_type begin, Index_type end,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               begin, end);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
           blockdata[BLOCK_LENGTH*k] += blockdata[(BLOCK_LENGTH*k)+i];
       }
    }

   RAJA_FT_END ;
}


//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over range index sets.
//
////////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over range segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                               begin, end);
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
 * \brief  Forall execution over range segment object with index count
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cuda_exec,
                   const RangeSegment& iseg,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_Icount_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                                      begin, end,
                                                      icount);
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
 * \brief  Forall min reduction over range segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(cuda_exec,
                const RangeSegment& iseg,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               begin, end);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata = 
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
          blockdata[BLOCK_LENGTH*k] =
             RAJA_MIN(blockdata[BLOCK_LENGTH*k],
                        blockdata[(BLOCK_LENGTH*k)+i]) ;
       }
    }

   RAJA_FT_END ;
}


//
// RDH -- need minloc reduction
//


/*!
 ******************************************************************************
 *
 * \brief  Forall max reduction over range segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_max(cuda_exec,
                const RangeSegment& iseg,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               begin, end);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata = 
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
          blockdata[BLOCK_LENGTH*k] =
             RAJA_MAX(blockdata[BLOCK_LENGTH*k],
                        blockdata[(BLOCK_LENGTH*k)+i]) ;
       }
    }

   RAJA_FT_END ;
}


//
// RDH -- need maxloc reduction
//


/*!
 ******************************************************************************
 *
 * \brief  Forall sum reduction over range segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cuda_exec,
                const RangeSegment& iseg,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               begin, end);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
           blockdata[BLOCK_LENGTH*k] += blockdata[(BLOCK_LENGTH*k)+i];
       }
    }

   RAJA_FT_END ;
}


//
////////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays. 
//
////////////////////////////////////////////////////////////////////////
//

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
            const Index_type*, Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (len + blockSize - 1) / blockSize;
   forall_kernel<<<gridSize, blockSize>>>(loop_body, 
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

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over indirection array with index count
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cuda_exec,
                   const Index_type* idx, Index_type len,
                   Index_type icount,
                   LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_Icount_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                                      idx, len,
                                                      icount);
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
 * \brief  Forall min reduction over indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(cuda_exec,
                const Index_type* idx, Index_type len,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for(int k=0;k<=cuda_reduction_gid;k++) {
        for (int i=1; i<=gridSize ; ++i) {
            blockdata[BLOCK_LENGTH*k] =
                RAJA_MIN(blockdata[BLOCK_LENGTH*k],
                           blockdata[(BLOCK_LENGTH*k)+i]) ;
        }
    }

   RAJA_FT_END ;
}


//
// RDH -- need minloc reduction
//


/*!
 ******************************************************************************
 * 
 * \brief  Forall max reduction over indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_max(cuda_exec,
                const Index_type* idx, Index_type len,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for(int k=0;k<=cuda_reduction_gid;k++) {
        for (int i=1; i<=gridSize ; ++i) {
            blockdata[BLOCK_LENGTH*k] =
                RAJA_MAX(blockdata[BLOCK_LENGTH*k],
                           blockdata[(BLOCK_LENGTH*k)+i]) ;
        }
    }

   RAJA_FT_END ;
}


//
// RDH -- need maxloc reduction
//


/*!
 ******************************************************************************
 *
 * \brief  Forall sum reduction over indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cuda_exec,
                const Index_type* idx, Index_type len,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
           blockdata[BLOCK_LENGTH*k] += blockdata[(BLOCK_LENGTH*k)+i];
       }
    }

   RAJA_FT_END ;
}


//
////////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segments.
//
////////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Forall execution for list segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec,
            const ListSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type* idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (len + blockSize - 1) / blockSize;
   forall_kernel<<<gridSize, blockSize>>>(loop_body, 
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

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over list segment object with index count
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cuda_exec,
                   const ListSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type* idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_Icount_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                                      idx, len,
                                                      icount);
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
 * \brief  Forall min reduction over list segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(cuda_exec,
                const ListSegment& iseg,
                LOOP_BODY loop_body)
{
   const Index_type* idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for(int k=0;k<=cuda_reduction_gid;k++) {
        for (int i=1; i<=gridSize ; ++i) {
            blockdata[BLOCK_LENGTH*k] =
                RAJA_MIN(blockdata[BLOCK_LENGTH*k],
                           blockdata[(BLOCK_LENGTH*k)+i]) ;
        }
    }

   RAJA_FT_END ;
}


//
// RDH -- need minloc reduction
//


/*!
 ******************************************************************************
 * 
 * \brief  Forall max reduction over list segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_max(cuda_exec,
                const ListSegment& iseg,
                LOOP_BODY loop_body)
{
   const Index_type* idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for(int k=0;k<=cuda_reduction_gid;k++) {
        for (int i=1; i<=gridSize ; ++i) {
            blockdata[BLOCK_LENGTH*k] =
                RAJA_MAX(blockdata[BLOCK_LENGTH*k],
                           blockdata[(BLOCK_LENGTH*k)+i]) ;
        }
    }

   RAJA_FT_END ;
}


//
// RDH -- need maxloc reduction
//


/*!
 ******************************************************************************
 *
 * \brief  Forall sum reduction over list segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cuda_exec,
                const ListSegment& iseg,
                LOOP_BODY loop_body)
{
   const Index_type* idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (end - begin + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body,
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

    CudaReduceBlockAllocType* blockdata =
       getReductionMemBlockData< CudaReduceBlockAllocType >() ;

    for (int k=0; k<=cuda_reduction_gid; k++) {
       for (int i=1; i<=gridSize; ++i) {
           blockdata[BLOCK_LENGTH*k] += blockdata[(BLOCK_LENGTH*k)+i];
       }
    }

   RAJA_FT_END ;
}


}  // closing brace for RAJA namespace


#endif  // if defined(RAJA_USE_CUDA)

#endif  // closing endif for header file include guard
