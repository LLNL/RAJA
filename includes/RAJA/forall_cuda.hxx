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

#include "MemUtils.hxx"

#include <iostream>
#include <cstdlib>

#include <cfloat>


namespace RAJA {

//
// Operations in this file are parametrized using the following
// values.  RDH -- should we move these somewhere else??
//
const int THREADS_PER_BLOCK = 256;
const int WARP_SIZE = 32;


//
//////////////////////////////////////////////////////////////////////
//
// Utility methods used in GPU reductions.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Method to shuffle 32b registers in sum reduction.
 *
 ******************************************************************************
 */
__device__ __forceinline__
double shfl_xor(double var, int laneMask)
{
    int lo = __shfl_xor( __double2loint(var), laneMask );
    int hi = __shfl_xor( __double2hiint(var), laneMask );
    return __hiloint2double( hi, lo );
}
 
/*!
 ******************************************************************************
 *
 * \brief Atomic min update method used to gather current min reduction value.
 *
 ******************************************************************************
 */
__device__ inline void atomicMin1(double *address, double value)
{
    unsigned long long oldval, newval, readback;
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(value);
    while ((readback = 
            atomicCAS((unsigned long long*)address,oldval,newval)) != oldval)
    {
        oldval = readback;
        newval = __double_as_longlong( 
                    RAJA_MIN(__longlong_as_double(oldval),value) );
    }
}
/// Alternative implementation that appears to be more robust (why? no one knows.)
__device__ inline void atomicMin2(double *address, double value)
{
    unsigned long long int* address_as_ull =
                            (unsigned long long int*)address;
    unsigned long long int oldval = *address_as_ull, assumed;

    do {
       assumed = oldval;
       oldval = atomicCAS( address_as_ull, assumed, __double_as_longlong( 
                        RAJA_MIN( __longlong_as_double(assumed), value) ) );
    } while (assumed != oldval);
}

/*!
 ******************************************************************************
 *
 * \brief Atomic min update method used to gather current min reduction value.
 *
 ******************************************************************************
 */
__device__ inline void atomicMax1(double *address, double value)
{
    unsigned long long oldval, newval, readback;
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(value);
    while ((readback =
            atomicCAS((unsigned long long*)address,oldval,newval)) != oldval)
    {
        oldval = readback;
        newval = __double_as_longlong(
                    RAJA_MAX(__longlong_as_double(oldval),value) );
    }
}
/// Alternative implementation that appears to be more robust (why? no one knows.)
__device__ inline void atomicMax2(double *address, double value)
{
    unsigned long long int* address_as_ull =
                            (unsigned long long int*)address;
    unsigned long long int oldval = *address_as_ull, assumed;

    do {
       assumed = oldval;
       oldval = atomicCAS( address_as_ull, assumed, __double_as_longlong(
                        RAJA_MAX( __longlong_as_double(assumed), value) ) );
    } while (assumed != oldval);
}



//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
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
class ReduceMin<cuda_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   // Ctor only executes on the host.
   //
   explicit ReduceMin(T init_val)
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCudaReductionId();
//    std::cout << "ReduceMin id = " << m_myID << std::endl;

      m_blockdata = getCudaReductionMemBlock() ;
      m_blockoffset = getCudaReductionMemBlockOffset(m_myID);
      m_blockdata[m_blockoffset] = init_val;

      cudaDeviceSynchronize();
   }

   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ ReduceMin( const ReduceMin<cuda_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ ~ReduceMin<cuda_reduce, T>()
   {
      if (!m_is_copy) {
#if defined( __CUDA_ARCH__ ) 
#else
         releaseCudaReductionId(m_myID); 
#endif
         // OK to perform cudaFree of cudaMalloc vars if needed...
      }
   }

   //
   // Operator to retrieve reduced min value (before object is destroyed).
   // Accessor only operates on host.
   //
   operator T()
   {
      cudaDeviceSynchronize() ;

      size_t current_grid_size = getCurrentGridSize();
      for (int i=1; i<=current_grid_size; ++i) {
         m_blockdata[m_blockoffset] =
             RAJA_MIN(m_blockdata[m_blockoffset],
                      m_blockdata[m_blockoffset+i]) ;
      }
      m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset]);

      return m_reduced_val;
   }

   //
   // Updates reduced value in the proper shared memory block locations.
   //
   __device__ ReduceMin<cuda_reduce, T> min(T val) const
   {
      __shared__ T sd[THREADS_PER_BLOCK];

      if (threadIdx.x == 0) {
         for(int i=0;i<THREADS_PER_BLOCK;i++) sd[i] = m_reduced_val;
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
#if 0
          atomicMin1( &m_blockdata[m_blockoffset],
#else
          atomicMin2( &m_blockdata[m_blockoffset],
#endif
                     RAJA_MIN( m_blockdata[m_blockoffset],
                              m_blockdata[m_blockoffset + blockIdx.x+1] ) );
      }

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMin<cuda_reduce, T>();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CudaReductionBlockDataType* m_blockdata;
   int m_blockoffset;
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
class ReduceMax<cuda_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   // Ctor only executes on the host.
   //
   explicit ReduceMax(T init_val)
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCudaReductionId();
//    std::cout << "ReduceMax id = " << m_myID << std::endl;

      m_blockdata = getCudaReductionMemBlock() ;
      m_blockoffset = getCudaReductionMemBlockOffset(m_myID);
      m_blockdata[m_blockoffset] = init_val;

      cudaDeviceSynchronize();
   }

   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ ReduceMax( const ReduceMax<cuda_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ ~ReduceMax<cuda_reduce, T>()
   {
      if (!m_is_copy) {
#if defined( __CUDA_ARCH__ )
#else
        releaseCudaReductionId(m_myID);
#endif
        // OK to perform cudaFree of cudaMalloc vars if needed...
      }
   }

   //
   // Operator to retrieve reduced max value (before object is destroyed).
   // Accessor only operates on host.
   //
   operator T()
   {
      cudaDeviceSynchronize() ;

      size_t current_grid_size = getCurrentGridSize();
      for (int i=1; i<=current_grid_size; ++i) {
         m_blockdata[m_blockoffset] =
             RAJA_MAX(m_blockdata[m_blockoffset],
                      m_blockdata[m_blockoffset+i]) ;
      }
      m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset]);

      return m_reduced_val;
   }

   //
   // Updates reduced value in the proper shared memory block locations.
   //
   __device__ ReduceMax<cuda_reduce, T> max(T val) const
   {
      __shared__ T sd[THREADS_PER_BLOCK];

      if (threadIdx.x == 0) {
         for(int i=0;i<THREADS_PER_BLOCK;i++) sd[i] = m_reduced_val;
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
#if 0
          atomicMax1( &m_blockdata[m_blockoffset],
#else
          atomicMax2( &m_blockdata[m_blockoffset],
#endif
                     RAJA_MAX( m_blockdata[m_blockoffset],
                              m_blockdata[m_blockoffset + blockIdx.x+1] ) );
      }

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMax<cuda_reduce, T>();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CudaReductionBlockDataType* m_blockdata;
   int m_blockoffset;
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
class ReduceSum<cuda_reduce, T> 
{
public:
   //
   // Constructor takes initial reduction value (default ctor is disabled).
   // Ctor only executes on the host.
   //
   explicit ReduceSum(T init_val)
   {
      m_is_copy = false;

      m_init_val = init_val;
      m_reduced_val = static_cast<T>(0);

      m_myID = getCudaReductionId();
//    std::cout << "ReduceSum id = " << m_myID << std::endl;

      m_blockdata = getCudaReductionMemBlock();
      m_blockoffset = getCudaReductionMemBlockOffset(m_myID);

      // Shared memory block must be initialized to zero so sum reduction is correct
      size_t len = getCudaReductionMemBlockOffset(m_myID+1) - m_blockoffset;
      cudaMemset(&m_blockdata[m_blockoffset], 0,
                 sizeof(CudaReductionBlockDataType)*len);

      cudaDeviceSynchronize();
   }

   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ ReduceSum( const ReduceSum<cuda_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ ~ReduceSum<cuda_reduce, T>()
   {
      if (!m_is_copy) {
#if defined( __CUDA_ARCH__ )
#else
         releaseCudaReductionId(m_myID);
#endif
         // OK to perform cudaFree of cudaMalloc vars if needed...
      }
   }

   //
   // Operator to retrieve reduced sum value (before object is destroyed).
   // Accessor only operates on host.
   //
   operator T()
   {
      cudaDeviceSynchronize() ;

      m_blockdata[m_blockoffset] = static_cast<T>(0);
      size_t current_grid_size = getCurrentGridSize();
      for (int i=1; i<=current_grid_size; ++i) {
         m_blockdata[m_blockoffset] += m_blockdata[m_blockoffset+i];
      }
      m_reduced_val = m_init_val + static_cast<T>(m_blockdata[m_blockoffset]);

      return m_reduced_val;
   }

   //
   // += operator to accumulate arg value in the proper shared
   // memory block location.
   //
   __device__ ReduceSum<cuda_reduce, T> operator+=(T val) const
   {
      __shared__ T sd[THREADS_PER_BLOCK];

      if (threadIdx.x == 0) {
         for(int i=0;i<THREADS_PER_BLOCK;i++) sd[i] = 0;
      }
      __syncthreads();

      sd[threadIdx.x] = val;

      T temp = 0;
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

      // one thread adds to gmem, we skip m_blockdata[m_blockoffset]
      // because we will be accumlating into this
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

   bool m_is_copy;
   int m_myID;

   T m_init_val;
   T m_reduced_val;

   CudaReductionBlockDataType* m_blockdata ;
   int m_blockoffset;
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
                                   Index_type begin, Index_type len)
{
   Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
   if (ii < len) {
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
                                          Index_type begin, Index_type len,
                                          Index_type icount)
{
   Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
   if (ii < len) {
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
                                          Index_type length,
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
   Index_type len = end - begin;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                               begin, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

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

   Index_type len = end - begin;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (len + blockSize - 1) / blockSize;
   forall_Icount_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                                      begin, len,
                                                      icount);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

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
   Index_type len = end - begin;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                               begin, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

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
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();

   RAJA_FT_BEGIN ;

   const Index_type len = iseg.getEnd() - begin;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (len + blockSize - 1) / blockSize;

   forall_Icount_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                                      begin, len,
                                                      icount);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

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
            const Index_type* idx, Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   size_t blockSize = THREADS_PER_BLOCK;
   size_t gridSize = (len + blockSize - 1) / blockSize;
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

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
   size_t gridSize = (len + blockSize - 1) / blockSize;
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

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

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
   forall_cuda_kernel<<<gridSize, blockSize>>>(loop_body, 
                                               idx, len);
#ifdef RAJA_SYNC
   if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
                << __LINE__ << std::endl;
      exit(1);
   }
#endif

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

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
   size_t gridSize = (len + blockSize - 1) / blockSize;
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

   // set current grid size for reductions that may have been done in forall...
   setCurrentGridSize(gridSize);

   RAJA_FT_END ;
}


}  // closing brace for RAJA namespace


#endif  // if defined(RAJA_USE_CUDA)

#endif  // closing endif for header file include guard
