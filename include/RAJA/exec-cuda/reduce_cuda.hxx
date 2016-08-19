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

#include <cassert>

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include "RAJA/exec-cuda/raja_cudaerrchk.hxx"


namespace RAJA
{


/*!
 ******************************************************************************
 *
 * \brief Method to shuffle 32b registers in sum reduction.
 *
 ******************************************************************************
 */
__device__ __forceinline__ double shfl_xor(double var, int laneMask)
{
  int lo = __shfl_xor(__double2loint(var), laneMask);
  int hi = __shfl_xor(__double2hiint(var), laneMask);
  return __hiloint2double(hi, lo);
}

// The following atomic functions need to be outside of the RAJA namespace
#include <cuda.h>

//
// Three different variants of min/max reductions can be run by choosing
// one of these macros. Only one should be defined!!!
//
//#define RAJA_USE_ATOMIC_ONE
#define RAJA_USE_ATOMIC_TWO
//#define RAJA_USE_NO_ATOMICS

#if 0
#define ull_to_double(x) __longlong_as_double(reinterpret_cast<long long>(x))

#define double_to_ull(x) \
  reinterpret_cast<unsigned long long>(__double_as_longlong(x))
#else
#define ull_to_double(x) __longlong_as_double(x)

#define double_to_ull(x) __double_as_longlong(x)
#endif

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
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp > value) {
    unsigned long long oldval, newval, readback;
    oldval = double_to_ull(temp);
    newval = double_to_ull(value);
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    while ((readback = atomicCAS(address_as_ull, oldval, newval)) != oldval) {
      oldval = readback;
      newval = double_to_ull(RAJA_MIN(ull_to_double(oldval), value));
    }
  }
}

///
__device__ inline void atomicMin(float *address, double value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp > value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = atomicCAS(address_as_i, oldval, newval)) != oldval) {
      oldval = readback;
      newval = __float_as_int(RAJA_MIN(__int_as_float(oldval), value));
    }
  }
}

///
__device__ inline void atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp < value) {
    unsigned long long oldval, newval, readback;
    oldval = double_to_ull(temp);
    newval = double_to_ull(value);
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    while ((readback = atomicCAS(address_as_ull, oldval, newval)) != oldval) {
      oldval = readback;
      newval = double_to_ull(RAJA_MAX(ull_to_double(oldval), value));
    }
  }
}

///
__device__ inline void atomicMax(float *address, double value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp < value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = atomicCAS(address_as_i, oldval, newval)) != oldval) {
      oldval = readback;
      newval = __float_as_int(RAJA_MAX(__int_as_float(oldval), value));
    }
  }
}

///
__device__ void atomicAdd(double *address, double value)
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = atomicCAS((unsigned long long *)address, oldval, newval))
         != oldval) {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
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
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp > value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = double_to_ull(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_ull,
                    assumed,
                    double_to_ull(RAJA_MIN(ull_to_double(assumed), value)));
    } while (assumed != oldval);
  }
}

///
__device__ inline void atomicMin(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp > value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MIN(__int_as_float(assumed), value)));
    } while (assumed != oldval);
  }
}

///
__device__ inline void atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp < value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = double_to_ull(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_ull,
                    assumed,
                    double_to_ull(RAJA_MAX(ull_to_double(assumed), value)));
    } while (assumed != oldval);
  }
}

///
__device__ inline void atomicMax(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp < value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MAX(__int_as_float(assumed), value)));
    } while (assumed != oldval);
  }
}

///
__device__ inline void atomicAdd(double *address, double value)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int oldval = *address_as_ull, assumed;

  do {
    assumed = oldval;
    oldval =
        atomicCAS(address_as_ull,
                  assumed,
                  __double_as_longlong(__longlong_as_double(oldval) + value));
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
class ReduceMin<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
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
  ReduceMin(const ReduceMin<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMin<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_val = static_cast<T>(m_tallydata->tally);
    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that updates min value in proper device memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMin<cuda_reduce<BLOCK_SIZE>, T> min(T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = m_reduced_val;
      }
    }
    __syncthreads();

    sd[threadId] = val;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      sd[0] = RAJA_MIN(sd[0], sd[1]);
      atomicMin(&(m_tallydata->tally), sd[0]);
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMin<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_reduced_val;

  CudaReductionBlockTallyType *m_tallydata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
};

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
class ReduceMax<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
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
  ReduceMax(const ReduceMax<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMax<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced max value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_val = static_cast<T>(m_tallydata->tally);
    return m_reduced_val;
  }

  //
  // Method that returns reduced max value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that updates max value in proper device memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMax<cuda_reduce<BLOCK_SIZE>, T> max(T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = m_reduced_val;
      }
    }
    __syncthreads();

    sd[threadId] = val;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      sd[0] = RAJA_MAX(sd[0], sd[1]);
      atomicMax(&(m_tallydata->tally), sd[0]);
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMax<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_reduced_val;

  CudaReductionBlockTallyType *m_tallydata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
};

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
class ReduceSum<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceSum(T init_val)
  {
    m_is_copy = false;

    m_init_val = init_val;
    m_reduced_val = static_cast<T>(0);

    m_myID = getCudaReductionId();
    //    std::cout << "ReduceSum id = " << m_myID << std::endl;

    m_blockdata = getCudaReductionMemBlock(m_myID);
    m_blockoffset = 1;

    // Entire global shared memory block must be initialized to zero so
    // sum reduction is correct.
    size_t len = RAJA_CUDA_REDUCE_BLOCK_LENGTH;
    cudaErrchk(cudaMemset(&m_blockdata[m_blockoffset],
                          0,
                          sizeof(CudaReductionBlockDataType) * len));

    m_max_grid_size = m_blockdata;
    m_max_grid_size[0] = 0;

    cudaErrchk(cudaDeviceSynchronize());
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceSum(const ReduceSum<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceSum<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());

    m_blockdata[m_blockoffset] = static_cast<T>(0);

    size_t grid_size = m_max_grid_size[0];
    assert(grid_size < RAJA_CUDA_REDUCE_BLOCK_LENGTH);
    for (size_t i = 1; i <= grid_size; ++i) {
      m_blockdata[m_blockoffset] += m_blockdata[m_blockoffset + i];
    }
    m_reduced_val = m_init_val + static_cast<T>(m_blockdata[m_blockoffset]);

    return m_reduced_val;
  }

  //
  // Method that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum in the proper device shared
  // memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceSum<cuda_reduce<BLOCK_SIZE>, T> operator+=(T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    if (blockId + threadId == 0) {
      m_max_grid_size[0] =
          RAJA_MAX(gridDim.x * gridDim.y * gridDim.z, m_max_grid_size[0]);
    }

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = m_reduced_val;
      }
    }
    __syncthreads();

    sd[threadId] = val;

    T temp = 0;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] += sd[threadId + i];
      }
      __syncthreads();
    }

    if (threadId < WARP_SIZE) {
      temp = sd[threadId];
      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        temp += shfl_xor(temp, i);
      }
    }

    // one thread adds to gmem, we skip m_blockdata[m_blockoffset]
    // because we will be accumlating into this
    if (threadId == 0) {
      m_blockdata[m_blockoffset + blockId + 1] += temp;
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;
  int m_myID;

  T m_init_val;
  T m_reduced_val;

  CudaReductionBlockDataType *m_blockdata;
  int m_blockoffset;

  CudaReductionBlockDataType *m_max_grid_size;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reduction Atomic Non-Deterministic Variant class template
 *         for use in CUDA kernel.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceSum(T init_val)
  {
    m_is_copy = false;
    m_reduced_val = static_cast<T>(0);
    m_init_val = init_val;
    m_myID = getCudaReductionId();
    m_tallydata = getCudaReductionTallyBlock(m_myID);
    m_tallydata->tally = static_cast<T>(0);
    m_tallydata->initVal = init_val;
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceSum(const ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  // Destruction on host releases the global shared memory block chunk for
  //
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaDeviceSynchronize();
    m_reduced_val = m_init_val + static_cast<T>(m_tallydata->tally);
    return m_reduced_val;
  }

  //
  // Operator that returns reduced sum value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum in the proper device
  // memory block locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T> operator+=(
      T val) const
  {
    __shared__ T sd[BLOCK_SIZE];

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i] = m_reduced_val;
      }
    }
    __syncthreads();

    sd[threadId] = val;

    T temp = 0;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] += sd[threadId + i];
      }
      __syncthreads();
    }

    if (threadId < WARP_SIZE) {
      temp = sd[threadId];
      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        temp += shfl_xor(temp, i);
      }
    }

    // one thread adds to tally
    if (threadId == 0) {
      atomicAdd(&(m_tallydata->tally), temp);
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<cuda_reduce_atomic<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;

  T m_init_val;
  T m_reduced_val;

  CudaReductionBlockTallyType *m_tallydata;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
};

///
/// Each ReduceMinLoc or ReduceMaxLoc object uses retiredBlocks as a way
/// to complete the reduction in a single pass. Although the algorithm
/// updates retiredBlocks via an atomicAdd(int) the actual reduction values
/// do not use atomics and require a finishing stage performed
/// by the last block.
///
__device__ __managed__ unsigned int retiredBlocks[RAJA_MAX_REDUCE_VARS];

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in a CUDA execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceMinLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;
    m_reduced_val = init_val;
    m_reduced_idx = init_loc;
    m_myID = getCudaReductionId();
    retiredBlocks[m_myID] = 0;
    m_blockdata = getCudaReductionLocMemBlock(m_myID);
    // we're adding max grid size calculation for an assert check in the
    // accessor
    m_max_grid_size = m_blockdata;
    m_max_grid_size[0].val = 0;
    m_blockoffset = 1;
    m_blockdata[m_blockoffset].val = init_val;
    m_blockdata[m_blockoffset].idx = init_loc;

    for (int j = 1; j <= RAJA_CUDA_REDUCE_BLOCK_LENGTH; ++j) {
      m_blockdata[m_blockoffset + j].val = init_val;
      m_blockdata[m_blockoffset + j].idx = init_loc;
    }
    cudaErrchk(cudaDeviceSynchronize());
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceMinLoc(const ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    size_t grid_size = m_max_grid_size[0].val;
    assert(grid_size < RAJA_CUDA_REDUCE_BLOCK_LENGTH);
    m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset].val);
    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that returns index value corresponding to the reduced min.
  //
  // Note: accessor only executes on host.
  //
  Index_type getLoc()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_idx = m_blockdata[m_blockoffset].idx;
    return m_reduced_idx;
  }

  //
  // Method that updates min and index values in proper device memory block
  // locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T> minloc(
      T val,
      Index_type idx) const
  {
    __shared__ CudaReductionLocBlockDataType sd[BLOCK_SIZE];
    __shared__ bool lastBlock;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    if (blockId + threadId == 0) {
      m_max_grid_size[0].val =
          RAJA_MAX(gridDim.x * gridDim.y * gridDim.z, m_max_grid_size[0].val);
    }

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i].val = m_reduced_val;
        sd[threadId + i].idx = m_reduced_idx;
      }
    }
    __syncthreads();

    sd[threadId].val = val;
    sd[threadId].idx = idx;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      lastBlock = false;
      sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 1]);
      m_blockdata[m_blockoffset + blockId + 1] =
          RAJA_MINLOC(sd[threadId], m_blockdata[m_blockoffset + blockId + 1]);
      __threadfence();
      unsigned int oldBlockCount =
          ::atomicAdd((unsigned int *)&retiredBlocks[m_myID], (unsigned int)1);
      lastBlock = (oldBlockCount == ((gridDim.x * gridDim.y * gridDim.z) - 1));
    }
    __syncthreads();

    if (lastBlock) {
      if (threadId == 0) {
        retiredBlocks[m_myID] = 0;
      }

      CudaReductionLocBlockDataType lmin = {m_reduced_val, m_reduced_idx};
      int blocks = gridDim.x * gridDim.y * gridDim.z;
      int threads = blockDim.x * blockDim.y * blockDim.z;
      for (int i = threadId; i < blocks; i += threads) {
        lmin = RAJA_MINLOC(lmin, m_blockdata[m_blockoffset + i + 1]);
      }
      sd[threadId] = lmin;
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + i]);
        }
        __syncthreads();
      }

      if (threadId < 16) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 16]);
      }
      __syncthreads();

      if (threadId < 8) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 8]);
      }
      __syncthreads();

      if (threadId < 4) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 4]);
      }
      __syncthreads();

      if (threadId < 2) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 2]);
      }
      __syncthreads();

      if (threadId < 1) {
        sd[threadId] = RAJA_MINLOC(sd[threadId], sd[threadId + 1]);
        m_blockdata[m_blockoffset] =
            RAJA_MINLOC(m_blockdata[m_blockoffset], sd[threadId]);
      }
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMinLoc<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;
  int m_blockoffset;

  T m_reduced_val;

  Index_type m_reduced_idx;

  CudaReductionLocBlockDataType *m_blockdata;
  CudaReductionLocBlockDataType *m_max_grid_size;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
};

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in a CUDA execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename T>
class ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T>
{
public:
  //
  // Constructor takes initial reduction value (default ctor is disabled).
  //
  // Note: Ctor only executes on the host.
  //
  explicit ReduceMaxLoc(T init_val, Index_type init_loc)
  {
    m_is_copy = false;
    m_reduced_val = init_val;
    m_reduced_idx = init_loc;
    m_myID = getCudaReductionId();
    retiredBlocks[m_myID] = 0;
    m_blockdata = getCudaReductionLocMemBlock(m_myID);
    // we're adding max grid size calculation for an assert check in the
    // accessor
    m_max_grid_size = m_blockdata;
    m_max_grid_size[0].val = 0;
    m_blockoffset = 1;
    m_blockdata[m_blockoffset].val = init_val;
    m_blockdata[m_blockoffset].idx = init_loc;

    for (int j = 1; j <= RAJA_CUDA_REDUCE_BLOCK_LENGTH; ++j) {
      m_blockdata[m_blockoffset + j].val = init_val;
      m_blockdata[m_blockoffset + j].idx = init_loc;
    }
    cudaErrchk(cudaDeviceSynchronize());
  }

  //
  // Copy ctor executes on both host and device.
  //
  __host__ __device__
  ReduceMaxLoc(const ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T> &other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction on host releases the global shared memory block chunk for
  // reduction id and id itself for others to use.
  //
  // Note: destructor executes on both host and device.
  //
  __host__ __device__ ~ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T>()
  {
    if (!m_is_copy) {
#if defined(__CUDA_ARCH__)
#else
      releaseCudaReductionId(m_myID);
#endif
    }
  }

  //
  // Operator that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    cudaErrchk(cudaDeviceSynchronize());
    size_t grid_size = m_max_grid_size[0].val;
    assert(grid_size < RAJA_CUDA_REDUCE_BLOCK_LENGTH);
    m_reduced_val = static_cast<T>(m_blockdata[m_blockoffset].val);
    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that returns index value corresponding to the reduced max.
  //
  // Note: accessor only executes on host.
  //
  Index_type getLoc()
  {
    cudaErrchk(cudaDeviceSynchronize());
    m_reduced_idx = m_blockdata[m_blockoffset].idx;
    return m_reduced_idx;
  }

  //
  // Method that updates max and index values in proper device memory block
  // locations.
  //
  // Note: only operates on device.
  //
  __device__ ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T> maxloc(
      T val,
      Index_type idx) const
  {
    __shared__ CudaReductionLocBlockDataType sd[BLOCK_SIZE];
    __shared__ bool lastBlock;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    if (blockId + threadId == 0) {
      m_max_grid_size[0].val =
          RAJA_MAX(gridDim.x * gridDim.y * gridDim.z, m_max_grid_size[0].val);
    }

    // initialize shared memory
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        // no need for __syncthreads()
        sd[threadId + i].val = m_reduced_val;
        sd[threadId + i].idx = m_reduced_idx;
      }
    }
    __syncthreads();

    sd[threadId].val = val;
    sd[threadId].idx = idx;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
      if (threadId < i) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + i]);
      }
      __syncthreads();
    }

    if (threadId < 16) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 16]);
    }
    __syncthreads();

    if (threadId < 8) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 8]);
    }
    __syncthreads();

    if (threadId < 4) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 4]);
    }
    __syncthreads();

    if (threadId < 2) {
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 2]);
    }
    __syncthreads();

    if (threadId < 1) {
      lastBlock = false;
      sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 1]);
      m_blockdata[m_blockoffset + blockId + 1] =
          RAJA_MAXLOC(sd[threadId], m_blockdata[m_blockoffset + blockId + 1]);
      __threadfence();
      unsigned int oldBlockCount =
          ::atomicAdd((unsigned int *)&retiredBlocks[m_myID], (unsigned int)1);
      lastBlock = (oldBlockCount == ((gridDim.x * gridDim.y * gridDim.z) - 1));
    }
    __syncthreads();

    if (lastBlock) {
      if (threadId == 0) {
        retiredBlocks[m_myID] = 0;
      }

      CudaReductionLocBlockDataType lmax = {m_reduced_val, m_reduced_idx};
      int blocks = gridDim.x * gridDim.y * gridDim.z;
      int threads = blockDim.x * blockDim.y * blockDim.z;

      for (int i = threadId; i < blocks; i += threads) {
        lmax = RAJA_MAXLOC(lmax, m_blockdata[m_blockoffset + i + 1]);
      }
      sd[threadId] = lmax;
      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + i]);
        }
        __syncthreads();
      }

      if (threadId < 16) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 16]);
      }
      __syncthreads();

      if (threadId < 8) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 8]);
      }
      __syncthreads();

      if (threadId < 4) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 4]);
      }
      __syncthreads();

      if (threadId < 2) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 2]);
      }
      __syncthreads();

      if (threadId < 1) {
        sd[threadId] = RAJA_MAXLOC(sd[threadId], sd[threadId + 1]);
        m_blockdata[m_blockoffset] =
            RAJA_MAXLOC(m_blockdata[m_blockoffset], sd[threadId]);
      }
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMaxLoc<cuda_reduce<BLOCK_SIZE>, T>();

  bool m_is_copy;

  int m_myID;
  int m_blockoffset;

  T m_reduced_val;

  Index_type m_reduced_idx;

  CudaReductionLocBlockDataType *m_blockdata;
  CudaReductionLocBlockDataType *m_max_grid_size;

  // Sanity checks for block size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
