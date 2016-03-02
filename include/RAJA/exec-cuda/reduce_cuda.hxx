/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for CUDA execution.
 *
 *          These methods should work on any platform that supports 
 *          CUDA devices.
 *
 ******************************************************************************
 */

#ifndef RAJA_reduce_cuda_HXX
#define RAJA_reduce_cuda_HXX

#include "RAJA/config.hxx"

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "MemUtils_CUDA.hxx"

#include <iostream>
#include <cstdlib>

#include <cfloat>


namespace RAJA {

//
// Three different variants of min/max reductions can be run by choosing
// one of these macros. Only one should be defined!!!
//
//#define RAJA_USE_ATOMIC_ONE
#define RAJA_USE_ATOMIC_TWO
//#define RAJA_USE_NO_ATOMICS

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

#if defined(RAJA_USE_ATOMIC_ONE)
/*!
 ******************************************************************************
 *
 * \brief Atomic min and max update methods used to gather current 
 *        min or max reduction value.
 *
 ******************************************************************************
 */
__device__ inline void atomicMin(double *address, double value)
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
///
__device__ inline void atomicMax(double *address, double value)
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

#elif defined(RAJA_USE_ATOMIC_TWO)

/*!
 ******************************************************************************
 *
 * \brief Alternative atomic min and max update methods used to gather current 
 *        min or max reduction value.
 *       
 *        These appear to be more robust than the ones above, not sure why.
 *
 ******************************************************************************
 */
__device__ inline void atomicMin(double *address, double value)
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
///
__device__ inline void atomicMax(double *address, double value)
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

#elif defined(RAJA_USE_NO_ATOMICS)

// Noting to do here...

#else

#error one of the options for using/not using atomics must be specified

#endif


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
 * \brief  Min reduction class template for use in CUDA kernels.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMin< cuda_reduce<BLOCK_SIZE>, T > 
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

      m_blockdata = getCudaReductionMemBlock(m_myID) ;
      m_blockoffset = 1;
      m_blockdata[m_blockoffset] = init_val;
#if defined(RAJA_USE_NO_ATOMICS)
      for (int j = 1; j <= RAJA_CUDA_REDUCE_BLOCK_LENGTH; ++j) {
         m_blockdata[m_blockoffset+j] = init_val;
      }
#endif

      m_max_grid_size = m_blockdata;
      m_max_grid_size[0] = 0;

      cudaDeviceSynchronize();
   }

   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ 
   ReduceMin( const ReduceMin< cuda_reduce<BLOCK_SIZE> , T >& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ 
   ~ReduceMin< cuda_reduce<BLOCK_SIZE> , T >()
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
#if defined(RAJA_USE_NO_ATOMICS)
      size_t grid_size = m_max_grid_size[0];
      for (size_t i=1; i <= grid_size; ++i) {
         m_blockdata[m_blockoffset] =
             RAJA_MIN(m_blockdata[m_blockoffset],
                      m_blockdata[m_blockoffset+i]) ;
      }
#endif
      m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset]);
      return m_reduced_val;
   }

   //
   // Updates reduced value in the proper shared memory block locations.
   //
   __device__ 
   ReduceMin< cuda_reduce<BLOCK_SIZE>, T > min(T val) const
   {
      __shared__ T sd[BLOCK_SIZE];

      if ( blockDim.x * blockIdx.x + threadIdx.x == 0 ) {
          m_max_grid_size[0] = RAJA_MAX( gridDim.x,  m_max_grid_size[0] );
      } 

      // initialize shared memory
      for ( int i = BLOCK_SIZE / 2; i > 0; i /=2 ) {     
          // this descends all the way to 1
          if ( threadIdx.x < i ) {                                
             // no need for __syncthreads()
             sd[threadIdx.x + i] = m_reduced_val;  
          } 
      }
      __syncthreads();


      sd[threadIdx.x] = val;
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
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
#if defined(RAJA_USE_NO_ATOMICS)
          m_blockdata[m_blockoffset + blockIdx.x+1]  = 
              RAJA_MIN( sd[threadIdx.x], 
                        m_blockdata[m_blockoffset + blockIdx.x+1] );
          
#else
          m_blockdata[m_blockoffset + blockIdx.x+1]  = sd[threadIdx.x];
          atomicMin( &m_blockdata[m_blockoffset],
                     RAJA_MIN( m_blockdata[m_blockoffset],
                               m_blockdata[m_blockoffset + blockIdx.x+1] ) );
#endif
      }

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMin< cuda_reduce<BLOCK_SIZE>, T >();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CudaReductionBlockDataType* m_blockdata;
   int m_blockoffset;

   CudaReductionBlockDataType* m_max_grid_size;
} ;


/*!
 ******************************************************************************
 *
 * \brief  Max reduction class template for use in CUDA kernels.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMax< cuda_reduce<BLOCK_SIZE>, T > 
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

      m_blockdata = getCudaReductionMemBlock(m_myID) ;
      m_blockoffset = 1;
      m_blockdata[m_blockoffset] = init_val;
#if defined(RAJA_USE_NO_ATOMICS)
      for (int j = 1; j <= RAJA_CUDA_REDUCE_BLOCK_LENGTH; ++j) {
         m_blockdata[m_blockoffset+j] = init_val;
      }
#endif

      m_max_grid_size = m_blockdata;
      m_max_grid_size[0] = 0;

      cudaDeviceSynchronize();
   }

   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ 
   ReduceMax( const ReduceMax< cuda_reduce<BLOCK_SIZE>, T >& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ 
   ~ReduceMax< cuda_reduce<BLOCK_SIZE>, T >()
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
#if defined(RAJA_USE_NO_ATOMICS)
      size_t grid_size = m_max_grid_size[0];
      for (size_t i = 1; i <= grid_size; ++i) {
         m_blockdata[m_blockoffset] =
             RAJA_MAX(m_blockdata[m_blockoffset],
                      m_blockdata[m_blockoffset+i]) ;
      }
#endif
      m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset]);

      return m_reduced_val;
   }

   //
   // Updates reduced value in the proper shared memory block locations.
   //
   __device__ 
   ReduceMax< cuda_reduce<BLOCK_SIZE>, T > max(T val) const
   {
      __shared__ T sd[BLOCK_SIZE];

      if ( blockDim.x * blockIdx.x + threadIdx.x == 0 ) {
         m_max_grid_size[0] = RAJA_MAX( gridDim.x,  m_max_grid_size[0] );
      }

       // initialize shared memory
      for ( int i = BLOCK_SIZE / 2; i > 0; i /=2 ) {     
          // this descends all the way to 1
          if ( threadIdx.x < i ) {
              // no need for __syncthreads()
              sd[threadIdx.x + i] = m_reduced_val;  
          } 
      }
      __syncthreads();

      sd[threadIdx.x] = val;
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
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
#if defined(RAJA_USE_NO_ATOMICS)
          m_blockdata[m_blockoffset + blockIdx.x+1]  =
              RAJA_MAX( sd[threadIdx.x],
                        m_blockdata[m_blockoffset + blockIdx.x+1] );

#else
          m_blockdata[m_blockoffset + blockIdx.x+1]  = sd[threadIdx.x];
          atomicMax( &m_blockdata[m_blockoffset],
                     RAJA_MAX( m_blockdata[m_blockoffset],
                               m_blockdata[m_blockoffset + blockIdx.x+1] ) );
#endif
      }

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMax< cuda_reduce<BLOCK_SIZE> , T >();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CudaReductionBlockDataType* m_blockdata;
   int m_blockoffset;

   CudaReductionBlockDataType* m_max_grid_size;
} ;


/*!
 ******************************************************************************
 *
 * \brief  Sum reduction class template for use in CUDA kernel.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceSum< cuda_reduce<BLOCK_SIZE>, T > 
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

      m_blockdata = getCudaReductionMemBlock(m_myID) ;
      m_blockoffset = 1;
      
      // Entire shared memory block must be initialized to zero so
      // sum reduction is correct.
      size_t len = RAJA_CUDA_REDUCE_BLOCK_LENGTH;
      cudaMemset(&m_blockdata[m_blockoffset], 0,
                 sizeof(CudaReductionBlockDataType)*len); 

      m_max_grid_size = m_blockdata;
      m_max_grid_size[0] = 0;

      cudaDeviceSynchronize();
   }

   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ 
   ReduceSum( const ReduceSum< cuda_reduce<BLOCK_SIZE>, T >& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ 
   ~ReduceSum< cuda_reduce<BLOCK_SIZE>, T >()
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

      size_t grid_size = m_max_grid_size[0];
      for (size_t i=1; i <= grid_size; ++i) {
         m_blockdata[m_blockoffset] += m_blockdata[m_blockoffset+i];
      }
      m_reduced_val = m_init_val + static_cast<T>(m_blockdata[m_blockoffset]);

      return m_reduced_val;
   }

   //
   // += operator to accumulate arg value in the proper shared
   // memory block location.
   //
   __device__ 
   ReduceSum<  cuda_reduce<BLOCK_SIZE>, T > operator+=(T val) const
   {
      __shared__ T sd[BLOCK_SIZE];

      if ( blockDim.x * blockIdx.x + threadIdx.x == 0 ) {
         m_max_grid_size[0] = RAJA_MAX( gridDim.x,  m_max_grid_size[0] );
      }

       // initialize shared memory
      for ( int i = BLOCK_SIZE / 2; i > 0; i /=2 ) {     
          // this descends all the way to 1
          if ( threadIdx.x < i ) {
              // no need for __syncthreads()
              sd[threadIdx.x + i] = m_reduced_val;  
          } 
      }
      __syncthreads();

      sd[threadIdx.x] = val;

      T temp = 0;
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
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
   ReduceSum< cuda_reduce<BLOCK_SIZE>, T >();

   bool m_is_copy;
   int m_myID;

   T m_init_val;
   T m_reduced_val;

   CudaReductionBlockDataType* m_blockdata ;
   int m_blockoffset;

   CudaReductionBlockDataType* m_max_grid_size;
} ;


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
