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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include "RAJA/exec-cuda/raja_cudaerrchk.hxx"


namespace RAJA {

//
// Three different variants of min/max reductions can be run by choosing
// one of these macros. Only one should be defined!!!
//
//#define RAJA_USE_ATOMIC_ONE
#define RAJA_USE_ATOMIC_TWO
//#define RAJA_USE_NO_ATOMICS

#if 0
#define ull_to_double(x) \
   __longlong_as_double(reinterpret_cast<long long>(x))

#define double_to_ull(x) \
   reinterpret_cast<unsigned long long>(__double_as_longlong(x))
#else
#define ull_to_double(x) __longlong_as_double(x)

#define double_to_ull(x) __double_as_longlong(x)
#endif


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
  double temp = *(reinterpret_cast<double volatile *>(address)) ;
  if (temp > value) {
    unsigned long long oldval, newval, readback;
    oldval = double_to_ull(temp);
    newval = double_to_ull(value);
    unsigned long long *address_as_ull =
             reinterpret_cast<unsigned long long *>(address);

    while ((readback = 
            atomicCAS(address_as_ull, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = double_to_ull(RAJA_MIN(ull_to_double(oldval), value)) ;
    }
  }
}

///
__device__ inline void atomicMin(float *address, double value)
{
  float temp = *(reinterpret_cast<float volatile *>(address)) ;
  if (temp > value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = 
            atomicCAS(address_as_i, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __float_as_int(RAJA_MIN(__int_as_float(oldval), value)) ;
    }
  }
}

///
__device__ inline void atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address)) ;
  if (temp < value) {
    unsigned long long oldval, newval, readback ;
    oldval = double_to_ull(temp);
    newval = double_to_ull(value);
    unsigned long long *address_as_ull =
             reinterpret_cast<unsigned long long *>(address);

    while ((readback =
            atomicCAS(address_as_ull, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = double_to_ull(RAJA_MAX(ull_to_double(oldval), value));
    }
  }
}

///
__device__ inline void atomicMax(float *address, double value)
{
  float temp = *(reinterpret_cast<float volatile *>(address)) ;
  if (temp < value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = 
            atomicCAS(address_as_i, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __float_as_int(RAJA_MAX(__int_as_float(oldval), value)) ;
    }
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
  double temp = *(reinterpret_cast<double volatile *>(address)) ;
  if (temp > value) {
    unsigned long long *address_as_ull =
             reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed ;
    unsigned long long oldval = double_to_ull(temp) ;
    do {
      assumed = oldval;
      oldval = atomicCAS(address_as_ull, assumed,
                         double_to_ull(RAJA_MIN(ull_to_double(assumed), value))
                        );
    } while (assumed != oldval);
  }
}
///
__device__ inline void atomicMin(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if(temp > value) {
    int *address_as_i = (int *) address;
    int assumed ;
    int oldval = __float_as_int(temp) ;
    do
    {
      assumed = oldval;
      oldval = atomicCAS(address_as_i, assumed,
                         __float_as_int(RAJA_MIN(__int_as_float(assumed),value))
                        );
    } while(assumed != oldval) ;
  }
}
///
__device__ inline void atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address)) ;
  if (temp < value) {
    unsigned long long *address_as_ull =
             reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed ;
    unsigned long long oldval = double_to_ull(temp) ;
    do {
      assumed = oldval;
      oldval = atomicCAS(address_as_ull, assumed,
                         double_to_ull(RAJA_MAX(ull_to_double(assumed), value))
                        );
    } while (assumed != oldval);
  }
}
///
__device__ inline void atomicMax(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if(temp < value) {
    int *address_as_i = (int *) address;
    int assumed ;
    int oldval = __float_as_int(temp) ;
    do
    {
      assumed = oldval;
      oldval = atomicCAS(address_as_i, assumed,
                         __float_as_int(RAJA_MAX(__int_as_float(assumed),value))
                        );
    } while(assumed != oldval) ;
  }
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
    m_tallydata = getCudaReductionTallyBlock(m_myID);
    m_tallydata->tally = init_val;
    m_tallydata->initVal = init_val;
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
     cudaErrchk(cudaDeviceSynchronize());
     m_reduced_val = static_cast<T>(m_tallydata->tally);
     return m_reduced_val;
   }

   //
   // Updates reduced value in the proper shared memory block locations.
   //
   __device__ 
   ReduceMin< cuda_reduce<BLOCK_SIZE>, T > min(T val) const
   {
     __shared__ T sd[BLOCK_SIZE];

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
         sd[0] = RAJA_MIN(sd[0],sd[1]);
         atomicMin(&(m_tallydata->tally),sd[0]);    
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

    CudaReductionBlockTallyType* m_tallydata;
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
     m_tallydata = getCudaReductionTallyBlock(m_myID);
     m_tallydata->tally = init_val;
     m_tallydata->initVal = init_val;
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
     cudaErrchk(cudaDeviceSynchronize());
     m_reduced_val = static_cast<T>(m_tallydata->tally);
     return m_reduced_val;
   }

   //
   // Updates reduced value in the proper shared memory block locations.
   //
   __device__ 
   ReduceMax< cuda_reduce<BLOCK_SIZE>, T > max(T val) const
   {
      __shared__ T sd[BLOCK_SIZE];

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
          sd[0] = RAJA_MAX(sd[0],sd[1]);
          atomicMax(&(m_tallydata->tally), sd[0] );
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

   CudaReductionBlockTallyType* m_tallydata;
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
      cudaErrchk(cudaMemset(&m_blockdata[m_blockoffset], 0,
                           sizeof(CudaReductionBlockDataType)*len)); 

      m_max_grid_size = m_blockdata;
      m_max_grid_size[0] = 0;

      cudaErrchk(cudaDeviceSynchronize());
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
      cudaErrchk(cudaDeviceSynchronize()) ;

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



///
/// each reduce variable involved in either ReduceMinLoc or ReduceMaxLoc
/// uses retiredBlocks as a way to complete the reduction in a single pass
/// Although the algorithm updates retiredBlocks via an atomicAdd(int) the actual
/// reduction values do not use atomics and require a finsihing stage performed
/// by the last block
__device__ __managed__ int retiredBlocks[RAJA_MAX_REDUCE_VARS] ;



template <size_t BLOCK_SIZE, typename T>
class ReduceMinLoc< cuda_reduce<BLOCK_SIZE>, T > 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMinLoc(T init_val, Index_type init_loc)
   {
      m_is_copy = false;
      m_reduced_val = init_val;
      m_reduced_idx = init_loc;
      m_myID = getCudaReductionId();
      retiredBlocks[m_myID] = 0;
      m_blockdata = getCudaReductionLocMemBlock(m_myID) ;
      m_blockoffset = 1;
      m_blockdata[m_blockoffset].val = init_val;
      m_blockdata[m_blockoffset].idx = init_loc;

      for (int j = 1; j <= RAJA_CUDA_REDUCE_BLOCK_LENGTH; ++j) {
         m_blockdata[m_blockoffset+j].val = init_val;
         m_blockdata[m_blockoffset+j].idx = init_loc;

      }
      cudaErrchk(cudaDeviceSynchronize());
   }


   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ 
   ReduceMinLoc( const ReduceMinLoc< cuda_reduce<BLOCK_SIZE> , T >& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ 
   ~ReduceMinLoc< cuda_reduce<BLOCK_SIZE> , T >()
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
      cudaErrchk(cudaDeviceSynchronize()) ;
      m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset].val);
      return m_reduced_val;
   }

   //
   // Operator to retrieve index value of min (before object is destroyed).
   //
   Index_type getMinLoc()
   {
      cudaErrchk(cudaDeviceSynchronize()) ; //it would be good not to call this
      m_reduced_idx = m_blockdata[m_blockoffset].idx;
      return m_reduced_idx;
   }

   //
   // Min-loc function 
   //
   __device__
   ReduceMinLoc< cuda_reduce<BLOCK_SIZE>, T> minloc(T val, Index_type idx) const 
   {

      __shared__ CudaReductionLocBlockDataType sd[BLOCK_SIZE];
      __shared__ bool lastBlock;

      // initialize shared memory
      for ( int i = BLOCK_SIZE / 2; i > 0; i /=2 ) {     
          // this descends all the way to 1
          if ( threadIdx.x < i ) {                                
             // no need for __syncthreads()
             sd[threadIdx.x + i].val = m_reduced_val; 
             sd[threadIdx.x + i].idx = m_reduced_idx; 
          } 
      }
      __syncthreads();

      sd[threadIdx.x].val = val;
      sd[threadIdx.x].idx = idx; // need to reconcile loc vs idx naming
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
          if (threadIdx.x < i) {
              sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x + i]);
          }
          __syncthreads();
      }

      if (threadIdx.x < 16) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+16]);
      }
      __syncthreads();

      if (threadIdx.x < 8) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+8]);
      }
      __syncthreads();

      if (threadIdx.x < 4) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+4]);
      }
      __syncthreads();

      if (threadIdx.x < 2) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+2]);
      }
      __syncthreads();

      if (threadIdx.x < 1) {
          lastBlock = false;
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+1]);
          m_blockdata[m_blockoffset + blockIdx.x+1]  = 
              RAJA_MINLOC( sd[threadIdx.x], 
                        m_blockdata[m_blockoffset + blockIdx.x+1] );
          unsigned int oldBlockCount = atomicAdd(&retiredBlocks[m_myID],1);
          lastBlock = (oldBlockCount == (gridDim.x-1));

      }
      __syncthreads();

      if(lastBlock) {
        if(threadIdx.x == 0) {
          retiredBlocks[m_myID] = 0;
        }

        CudaReductionLocBlockDataType lmin;
        lmin.val = m_reduced_val;
        lmin.idx = m_reduced_idx;
        for (int i = threadIdx.x; i < gridDim.x; i += BLOCK_SIZE) {
            lmin = RAJA_MINLOC(lmin,m_blockdata[m_blockoffset+i+1]);
        }
        sd[threadIdx.x] = lmin;
        __syncthreads();

        for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
          if (threadIdx.x < i) {
              sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x + i]);
          }
          __syncthreads();
        }

        if (threadIdx.x < 16) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+16]);
        }
        __syncthreads();

        if (threadIdx.x < 8) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+8]);
        }
        __syncthreads();

        if (threadIdx.x < 4) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+4]);
        }
        __syncthreads();

        if (threadIdx.x < 2) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+2]);
        }
        __syncthreads();

        if (threadIdx.x < 1) {
          sd[threadIdx.x] = RAJA_MINLOC(sd[threadIdx.x],sd[threadIdx.x+1]);
          m_blockdata[m_blockoffset] = RAJA_MINLOC(m_blockdata[m_blockoffset],sd[threadIdx.x]);
        }
      }
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMinLoc< cuda_reduce<BLOCK_SIZE>, T >();

   bool m_is_copy;

   int m_myID;
   int m_blockoffset;

   T m_reduced_val;

   Index_type m_reduced_idx;

   CudaReductionLocBlockDataType* m_blockdata;
} ;


template <size_t BLOCK_SIZE, typename T>
class ReduceMaxLoc< cuda_reduce<BLOCK_SIZE>, T > 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMaxLoc(T init_val, Index_type init_loc)
   {
      m_is_copy = false;
      m_reduced_val = init_val;
      m_reduced_idx = init_loc;
      m_myID = getCudaReductionId();
      retiredBlocks[m_myID] = 0;
      m_blockdata = getCudaReductionLocMemBlock(m_myID) ;
      m_blockoffset = 1;
      m_blockdata[m_blockoffset].val = init_val;
      m_blockdata[m_blockoffset].idx = init_loc;

      for (int j = 1; j <= RAJA_CUDA_REDUCE_BLOCK_LENGTH; ++j) {
         m_blockdata[m_blockoffset+j].val = init_val;
         m_blockdata[m_blockoffset+j].idx = init_loc;

      }
      cudaErrchk(cudaDeviceSynchronize());
   }


   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ 
   ReduceMaxLoc( const ReduceMaxLoc< cuda_reduce<BLOCK_SIZE> , T >& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ 
   ~ReduceMaxLoc< cuda_reduce<BLOCK_SIZE> , T >()
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
      cudaErrchk(cudaDeviceSynchronize()) ;
      m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset].val);
      return m_reduced_val;
   }

   //
   // Operator to retrieve index value of min (before object is destroyed).
   //
   Index_type getMaxLoc()
   {
      cudaErrchk(cudaDeviceSynchronize()) ; //it would be good not to call this
      m_reduced_idx = m_blockdata[m_blockoffset].idx;
      return m_reduced_idx;
   }

   //
   // Max-loc function 
   //
   __device__
   ReduceMaxLoc< cuda_reduce<BLOCK_SIZE>, T> maxloc(T val, Index_type idx) const 
   {

      __shared__ CudaReductionLocBlockDataType sd[BLOCK_SIZE];
      __shared__ bool lastBlock;

      // initialize shared memory
      for ( int i = BLOCK_SIZE / 2; i > 0; i /=2 ) {     
          // this descends all the way to 1
          if ( threadIdx.x < i ) {                                
             // no need for __syncthreads()
             sd[threadIdx.x + i].val = m_reduced_val; 
             sd[threadIdx.x + i].idx = m_reduced_idx; 
          } 
      }
      __syncthreads();

      sd[threadIdx.x].val = val;
      sd[threadIdx.x].idx = idx; // need to reconcile loc vs idx naming
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
          if (threadIdx.x < i) {
              sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x + i]);
          }
          __syncthreads();
      }

      if (threadIdx.x < 16) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+16]);
      }
      __syncthreads();

      if (threadIdx.x < 8) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+8]);
      }
      __syncthreads();

      if (threadIdx.x < 4) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+4]);
      }
      __syncthreads();

      if (threadIdx.x < 2) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+2]);
      }
      __syncthreads();

      if (threadIdx.x < 1) {
          lastBlock = false;
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+1]);
          m_blockdata[m_blockoffset + blockIdx.x+1]  = 
              RAJA_MAXLOC( sd[threadIdx.x], 
                        m_blockdata[m_blockoffset + blockIdx.x+1] );
          unsigned int oldBlockCount = atomicAdd(&retiredBlocks[m_myID],1);
          lastBlock = (oldBlockCount == (gridDim.x-1));

      }
      __syncthreads();

      if(lastBlock) {
        if(threadIdx.x == 0) {
          retiredBlocks[m_myID] = 0;
        }

        CudaReductionLocBlockDataType lmax;
        lmax.val = m_reduced_val;
        lmax.idx = m_reduced_idx;
        for (int i = threadIdx.x; i < gridDim.x; i += BLOCK_SIZE) {
            lmax = RAJA_MAXLOC(lmax,m_blockdata[m_blockoffset+i+1]);
        }
        sd[threadIdx.x] = lmax;
        __syncthreads();

        for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
          if (threadIdx.x < i) {
              sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x + i]);
          }
          __syncthreads();
        }

        if (threadIdx.x < 16) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+16]);
        }
        __syncthreads();

        if (threadIdx.x < 8) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+8]);
        }
        __syncthreads();

        if (threadIdx.x < 4) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+4]);
        }
        __syncthreads();

        if (threadIdx.x < 2) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+2]);
        }
        __syncthreads();

        if (threadIdx.x < 1) {
          sd[threadIdx.x] = RAJA_MAXLOC(sd[threadIdx.x],sd[threadIdx.x+1]);
          m_blockdata[m_blockoffset] = RAJA_MAXLOC(m_blockdata[m_blockoffset],sd[threadIdx.x]);
        }
      }
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMaxLoc< cuda_reduce<BLOCK_SIZE>, T >();

   bool m_is_copy;

   int m_myID;
   int m_blockoffset;

   T m_reduced_val;

   Index_type m_reduced_idx;

   CudaReductionLocBlockDataType* m_blockdata;
} ;

}  // closing brace for RAJA namespace


#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
