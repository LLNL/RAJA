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

#include <cuda.h>


namespace RAJA
{

// unnamed namespace to encapsulate helper functions
namespace
{
/*!
 ******************************************************************************
 *
 * \brief Method to shuffle 32b registers in sum reduction for arbitrary type.
 *
 ******************************************************************************
 */
template<typename T>
__device__ __forceinline__ T shfl_xor(T var, int laneMask)
{
  const int int_sizeof_T = 
      (sizeof(T) + sizeof(int) - 1) / sizeof(int);
  union {
    T var;
    int arr[int_sizeof_T];
  } Tunion;
  Tunion.var = var;

  for(int i = 0; i < int_sizeof_T; ++i) {
    Tunion.arr[i] = __shfl_xor(Tunion.arr[i], laneMask);
  }
  return Tunion.var;
}


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

/*!
 ******************************************************************************
 *
 * \brief Generics of atomic update methods used in reduction variables.
 *
 * The generic version just wraps the nvidia cuda atomics.
 * Specializations implement other more atomics using atomic CAS.
 *
 ******************************************************************************
 */
template <typename T>
__device__ inline T _atomicMin(T *address, T value)
{
  return atomicMin(address, value);
}

template <typename T>
__device__ inline T _atomicMax(T *address, T value)
{
  return atomicMax(address, value);
}

template <typename T>
__device__ inline T _atomicAdd(T *address, T value)
{
  return atomicAdd(address, value);
}

//
// Template specializations for atomic update methods not defined by nvidia
// cuda.
//
#if defined(RAJA_USE_ATOMIC_ONE)
/*!
 ******************************************************************************
 *
 * \brief Atomic min and max update methods used to gather current
 *        min or max reduction value.
 *
 ******************************************************************************
 */
template <>
__device__ inline double _atomicMin(double *address, double value)
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
    temp = ull_to_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float _atomicMin(float *address, float value)
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
    temp = __int_as_float(oldval);
  }
  return temp;
}
///
template <>
__device__ inline double _atomicMax(double *address, double value)
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
    temp = ull_to_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float _atomicMax(float *address, float value)
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
    temp = __int_as_float(oldval);
  }
  return temp;
}
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
// don't specialize for 64-bit min/max if they exist
#else
// implement 64-bit min/max if they don't exist
template <>
__device__ inline unsigned long long int _atomicMin(
                                  unsigned long long int *address,
                                  unsigned long long int value)
{
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp > value) {
    unsigned long long int oldval, newval;
    oldval = temp;
    newval = value;

    while ((readback = atomicCAS(address, oldval, newval)) != oldval) {
      oldval = readback;
      newval = RAJA_MIN(oldval, value);
    }
  }
  return readback;
}
///
template <>
__device__ inline unsigned long long int _atomicMax(
                                  unsigned long long int *address,
                                  unsigned long long int value)
{
  unsigned long long int readback =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (readback < value) {
    unsigned long long int oldval, newval;
    oldval = readback;
    newval = value;

    while ((readback = atomicCAS(address, oldval, newval)) != oldval) {
      oldval = readback;
      newval = RAJA_MAX(oldval, value);
    }
  }
  return readback;
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// don't specialize for 64-bit add if it exists
#else
///
template <>
__device__ inline double _atomicAdd(double *address, double value)
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = atomicCAS((unsigned long long *)address, oldval, newval))
         != oldval) {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
  return __longlong_as_double(oldval);
}
#endif

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
template <>
__device__ inline double _atomicMin(double *address, double value)
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
    temp = ull_to_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float _atomicMin(float *address, float value)
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
    temp = __int_as_float(oldval);
  }
  return temp;
}
///
template <>
__device__ inline double _atomicMax(double *address, double value)
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
    temp = ull_to_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float _atomicMax(float *address, float value)
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
    temp = __int_as_float(oldval);
  }
  return temp;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
// don't specialize for 64-bit min/max if they exist
#else
///
template <>
__device__ inline unsigned long long int _atomicMin(
                                  unsigned long long int *address,
                                  unsigned long long int value)
{
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp > value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval = atomicCAS(address, assumed, RAJA_MIN(assumed, value));
    } while (assumed != oldval);
    temp = oldval;
  }
  return temp;
}
///
template <>
__device__ inline unsigned long long int _atomicMax(
                                  unsigned long long int *address,
                                  unsigned long long int value)
{
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp < value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval = atomicCAS(address, assumed, RAJA_MAX(assumed, value));
    } while (assumed != oldval);
    temp = oldval;
  }
  return temp;
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// don't specialize for doubles if they exist
#else
///
template <>
__device__ inline double _atomicAdd(double *address, double value)
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
  return __longlong_as_double(oldval);
}
#endif

#elif defined(RAJA_USE_NO_ATOMICS)

// Noting to do here...

#else

#error one of the options for using/not using atomics must be specified

#endif

} // end unnamed namespace for helper functions

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
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>
{
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */

  explicit ReduceMin(T init_val)
  {
    m_is_copy_host = false;
    m_myID = getCudaReductionId();
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_tally_host,
                               (void **)&m_tally_device);
    m_tally_host->tally = init_val;
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  __host__ __device__
  ReduceMin(const ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
  {
    *this = other;
#if defined(__CUDA_ARCH__)
    m_is_copy_device = true;
    m_finish_reduction = !other.m_is_copy_device;
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    T val = m_tally_device->tally;
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        sd[threadId + i] = val;
      }
    }
    if (threadId < 1) {
      sd[threadId] = val;
    }
    __syncthreads();
#else
    m_is_copy_host = true;
    m_smem_offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  __host__ __device__ ~ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_finish_reduction) {
      extern __shared__ unsigned char sd_block[];
      T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + i]);
        }
        __syncthreads();
      }

      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MIN(sd[threadId], sd[threadId + i]);
        }
      }

      if (threadId < 1) {
        _atomicMin<T>(&m_tally_device->tally, sd[threadId]);
      }
    }
#else
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally;
  }

  /*!
   * \brief Method that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Method that updates min value.
   *
   * Note: only operates on device.
   */
  __device__ ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T> const &min(T val) const
  {
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    sd[threadId] = RAJA_MIN(sd[threadId], val);

    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>();

  /*!
   * \brief Pointer to host tally block cache slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_host = nullptr;

  /*!
   * \brief Pointer to device tally block slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_device = nullptr;

  /*!
   * \brief My cuda reduction variable ID.
   */
  int m_myID = -1;

  /*!
   * \brief Byte offset into dynamic shared memory.
   */
  int m_smem_offset = -1;

  /*!
   * \brief If this variable is a copy or not; only original may release memory 
   *        or perform finalization.
   */
  bool m_is_copy_host = false;
  bool m_is_copy_device = false;
  bool m_finish_reduction = false;

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       && (sizeof(CudaReductionBlockType<T>)
           <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
      "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
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
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>
{
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceMax(T init_val)
  {
    m_is_copy_host = false;
    m_myID = getCudaReductionId();
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_tally_host,
                               (void **)&m_tally_device);
    m_tally_host->tally = init_val;
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  __host__ __device__
  ReduceMax(const ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
  {
    *this = other;
#if defined(__CUDA_ARCH__)
    m_is_copy_device = true;
    m_finish_reduction = !other.m_is_copy_device;
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    T val = m_tally_device->tally;
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        sd[threadId + i] = val;
      }
    }
    if (threadId < 1) {
      sd[threadId] = val;
    }

    __syncthreads();
#else
    m_is_copy_host = true;
    m_smem_offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completed the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  __host__ __device__ ~ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_finish_reduction) {
      extern __shared__ unsigned char sd_block[];
      T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + i]);
        }
        __syncthreads();
      }

      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        if (threadId < i) {
          sd[threadId] = RAJA_MAX(sd[threadId], sd[threadId + i]);
        }
      }

      if (threadId < 1) {
        _atomicMax<T>(&m_tally_device->tally, sd[threadId]);
      }
    }
#else
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced max value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally;
  }

  /*!
   * \brief Method that returns reduced max value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Method that updates max value.
   *
   * Note: only operates on device.
   */
  __device__ ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T> const &max(T val)
   const
  {
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    sd[threadId] = RAJA_MAX(sd[threadId], val);

    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>();

  /*!
   * \brief Pointer to host tally block cache slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_host = nullptr;

  /*!
   * \brief Pointer to device tally block slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_device = nullptr;

  /*!
   * \brief My cuda reduction variable ID.
   */
  int m_myID = -1;

  /*!
   * \brief Byte offset into dynamic shared memory.
   */
  int m_smem_offset = -1;

  /*!
   * \brief If this variable is a copy or not; only original may release memory 
   *        or perform finalization.
   */
  bool m_is_copy_host = false;
  bool m_is_copy_device = false;
  bool m_finish_reduction = false;

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       && (sizeof(CudaReductionBlockType<T>)
           <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
      "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
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
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>
{
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceSum(T init_val)
  {
    m_is_copy_host = false;
    m_myID = getCudaReductionId();
    getCudaReductionMemBlock(m_myID, (void **)&m_blockdata);
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_tally_host,
                               (void **)&m_tally_device);
    m_tally_host->tally = init_val;
    m_tally_host->retiredBlocks = static_cast<GridSizeType>(0);
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  __host__ __device__
  ReduceSum(const ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
  {
    *this = other;
#if defined(__CUDA_ARCH__)
    m_is_copy_device = true;
    m_finish_reduction = !other.m_is_copy_device;
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    T val = static_cast<T>(0);
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        sd[threadId + i] = val;
      }
    }
    if (threadId < 1) {
      sd[threadId] = val;
    }

    __syncthreads();
#else
    m_is_copy_host = true;
    m_smem_offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  __host__ __device__ ~ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_finish_reduction) {
      extern __shared__ unsigned char sd_block[];
      T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

      int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;

      int blocks = gridDim.x * gridDim.y * gridDim.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] += sd[threadId + i];
        }
        __syncthreads();
      }

      T temp;
      if (threadId < WARP_SIZE) {
        temp = sd[threadId];
        for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
          temp += shfl_xor<T>(temp, i);
        }
      }

      bool lastBlock = false;
      if (threadId < 1) {
        // write data to global memory block
        m_blockdata->values[blockId] = temp;
        // ensure write visible to all threadblocks
        __threadfence();
        // increment counter, (wraps back to zero at second parameter)
        unsigned int oldBlockCount =
            atomicInc((unsigned int *)&m_tally_device->retiredBlocks,
                      (blocks - 1));
        lastBlock = (oldBlockCount == (blocks - 1));
      }

      // returns non-zero value if any thread passes in a non-zero value
      lastBlock = __syncthreads_or(lastBlock);

      if (lastBlock) {
        T temp = static_cast<T>(0);

        int threads = blockDim.x * blockDim.y * blockDim.z;
        for (int i = threadId; i < blocks; i += threads) {
          temp += m_blockdata->values[i];
        }
        // any unused slots were initialized in copy constructor
        sd[threadId] = temp;
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
            temp += shfl_xor<T>(temp, i);
          }
        }

        if (threadId < 1) {
          // add reduction to tally
          m_tally_device->tally += temp;
        }
      }
    }
#else
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally;
  }

  /*!
   * \brief Method that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Operator that adds value to sum.
   *
   * Note: only operates on device.
   */
  __device__ ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T> const &operator+=(
      T val) const
  {
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    sd[threadId] += val;


    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>();

  /*!
   * \brief Pointer to host tally block cache slot for this reduction variable.
   */
  CudaReductionTallyType<T> *m_tally_host = nullptr;

  /*!
   * \brief Pointer to device data block for this reduction variable.
   */
  CudaReductionBlockType<T> *m_blockdata = nullptr;

  /*!
   * \brief Pointer to device tally block slot for this reduction variable.
   */
  CudaReductionTallyType<T> *m_tally_device = nullptr;

  /*!
   * \brief My cuda reduction variable ID.
   */
  int m_myID = -1;

  /*!
   * \brief Byte offset into dynamic shared memory.
   */
  int m_smem_offset = -1;

  /*!
   * \brief If this variable is a copy or not; only original may release memory 
   *        or perform finalization.
   */
  bool m_is_copy_host = false;
  bool m_is_copy_device = false;
  bool m_finish_reduction = false;

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       && (sizeof(CudaReductionBlockType<T>)
           <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
      "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
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
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>
{
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceSum(T init_val)
  {
    m_is_copy_host = false;
    m_myID = getCudaReductionId();
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_tally_host,
                               (void **)&m_tally_device);
    m_tally_host->tally = init_val;
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  __host__ __device__
  ReduceSum(const ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T> &other)
  {
    *this = other;
#if defined(__CUDA_ARCH__)
    m_is_copy_device = true;
    m_finish_reduction = !other.m_is_copy_device;
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    T val = static_cast<T>(0);
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        sd[threadId + i] = val;
      }
    }
    if (threadId < 1) {
      sd[threadId] = val;
    }

    __syncthreads();
#else
    m_is_copy_host = true;
    m_smem_offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  __host__ __device__ ~ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_finish_reduction) {
      extern __shared__ unsigned char sd_block[];
      T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

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
          temp += shfl_xor<T>(temp, i);
        }
      }

      // one thread adds to tally
      if (threadId == 0) {
        _atomicAdd<T>(&(m_tally_device->tally), temp);
      }
    }
#else
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif

    
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally;
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Operator that adds value to sum.
   *
   * Note: only operates on device.
   */
  __device__ ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T> const &
  operator+=(T val) const
  {
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    sd[threadId] += val;

    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>();

  /*!
   * \brief Pointer to host tally block cache slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_host = nullptr;

  /*!
   * \brief Pointer to device tally block slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_device = nullptr;

  /*!
   * \brief My cuda reduction variable ID.
   */
  int m_myID = -1;

  /*!
   * \brief Byte offset into dynamic shared memory.
   */
  int m_smem_offset = -1;

  /*!
   * \brief If this variable is a copy or not; only original may release memory 
   *        or perform finalization.
   */
  bool m_is_copy_host = false;
  bool m_is_copy_device = false;
  bool m_finish_reduction = false;

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       && (sizeof(CudaReductionBlockType<T>)
           <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
      "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};


// set index first to avoid changes to val1 or val2 when writing to val_set
#define RAJA_CUDA_MINLOC(val_set, idx_set, val1, idx1, val2, idx2) \
  idx_set = ((val2) < (val1) ? (idx2) : (idx1));                   \
  val_set = ((val2) < (val1) ? (val2) : (val1));


#define RAJA_CUDA_MAXLOC(val_set, idx_set, val1, idx1, val2, idx2) \
  idx_set = ((val2) > (val1) ? (idx2) : (idx1));                   \
  val_set = ((val2) > (val1) ? (val2) : (val1));


/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in a CUDA execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>
{
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceMinLoc(T init_val, Index_type init_loc)
  {
    m_is_copy_host = false;
    m_myID = getCudaReductionId();
    getCudaReductionMemBlock(m_myID, (void **)&m_blockdata);
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_tally_host,
                               (void **)&m_tally_device);
    m_tally_host->tally.val = init_val;
    m_tally_host->tally.idx = init_loc;
    m_tally_host->retiredBlocks = static_cast<GridSizeType>(0);
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  __host__ __device__
  ReduceMinLoc(const ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
  {
    *this = other;
#if defined(__CUDA_ARCH__)
    m_is_copy_device = true;
    m_finish_reduction = !other.m_is_copy_device;
    extern __shared__ unsigned char sd_block[];
    T *sd_val = reinterpret_cast<T *>(&sd_block[m_smem_offset]);
    Index_type *sd_idx = reinterpret_cast<Index_type *>(
        &sd_block[m_smem_offset + sizeof(T) * BLOCK_SIZE]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    T val = m_tally_device->tally.val;
    Index_type idx = m_tally_device->tally.idx;
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        sd_val[threadId + i] = val;
        sd_idx[threadId + i] = idx;
      }
    }
    if (threadId < 1) {
      sd_val[threadId] = val;
      sd_idx[threadId] = idx;
    }

    __syncthreads();
#else
    m_is_copy_host = true;
    m_smem_offset =
        getCudaSharedmemOffset(m_myID, BLOCK_SIZE, 
                               (sizeof(T) + sizeof(Index_type)));
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  __host__ __device__ ~ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_finish_reduction) {
      extern __shared__ unsigned char sd_block[];
      T *sd_val = reinterpret_cast<T *>(&sd_block[m_smem_offset]);
      Index_type *sd_idx = reinterpret_cast<Index_type *>(
          &sd_block[m_smem_offset + sizeof(T) * BLOCK_SIZE]);

      int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;

      int blocks = gridDim.x * gridDim.y * gridDim.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          RAJA_CUDA_MINLOC(sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId + i],
                           sd_idx[threadId + i]);
        }
        __syncthreads();
      }

      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        if (threadId < i) {
          RAJA_CUDA_MINLOC(sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId + i],
                           sd_idx[threadId + i]);
        }
      }

      bool lastBlock = false;
      if (threadId < 1) {
        m_blockdata->values[blockId] = sd_val[threadId];
        m_blockdata->indices[blockId] = sd_idx[threadId];

        __threadfence();
        unsigned int oldBlockCount =
            atomicInc((unsigned int *)&m_tally_device->retiredBlocks,
                      (blocks - 1));
        lastBlock = (oldBlockCount == (blocks - 1));
      }
      lastBlock = __syncthreads_or(lastBlock);

      if (lastBlock) {
        CudaReductionLocType<T> lmin{sd_val[0], sd_idx[0]};

        int threads = blockDim.x * blockDim.y * blockDim.z;
        for (int i = threadId; i < blocks; i += threads) {
          RAJA_CUDA_MINLOC(lmin.val,
                           lmin.idx,
                           lmin.val,
                           lmin.idx,
                           m_blockdata->values[i],
                           m_blockdata->indices[i]);
        }
        sd_val[threadId] = lmin.val;
        sd_idx[threadId] = lmin.idx;
        __syncthreads();

        for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
          if (threadId < i) {
            RAJA_CUDA_MINLOC(sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId + i],
                             sd_idx[threadId + i]);
          }
          __syncthreads();
        }

        for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
          if (threadId < i) {
            RAJA_CUDA_MINLOC(sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId + i],
                             sd_idx[threadId + i]);
          }
        }

        if (threadId < 1) {
          RAJA_CUDA_MINLOC(m_tally_device->tally.val,
                           m_tally_device->tally.idx,
                           m_tally_device->tally.val,
                           m_tally_device->tally.idx,
                           sd_val[threadId],
                           sd_idx[threadId]);
        }
      }
    }
#else
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally.val;
  }

  /*!
   * \brief Method that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Method that returns index value corresponding to the reduced min.
   *
   * Note: accessor only executes on host.
   */
  Index_type getLoc()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally.idx;
  }

  /*!
   * \brief Method that updates min and index values.
   *
   * Note: only operates on device.
   */
  __device__ ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T> const &minloc(
      T val,
      Index_type idx) const
  {
    extern __shared__ unsigned char sd_block[];
    T *sd_val = reinterpret_cast<T *>(&sd_block[m_smem_offset]);
    Index_type *sd_idx = reinterpret_cast<Index_type *>(
        &sd_block[m_smem_offset + sizeof(T) * BLOCK_SIZE]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    RAJA_CUDA_MINLOC(sd_val[threadId],
                     sd_idx[threadId],
                     sd_val[threadId],
                     sd_idx[threadId],
                     val,
                     idx);

    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>();

  /*!
   * \brief Pointer to host tally block cache slot for this reduction variable.
   */
  CudaReductionLocTallyType<T> *m_tally_host = nullptr;

  /*!
   * \brief Pointer to device data block for this reduction variable.
   */
  CudaReductionLocBlockType<T> *m_blockdata = nullptr;

  /*!
   * \brief Pointer to device tally block slot for this reduction variable.
   */
  CudaReductionLocTallyType<T> *m_tally_device = nullptr;

  /*!
   * \brief My cuda reduction variable ID.
   */
  int m_myID = -1;

  /*!
   * \brief Byte offset into dynamic shared memory.
   */
  int m_smem_offset = -1;

  /*!
   * \brief If this variable is a copy or not; only original may release memory 
   *        or perform finalization.
   */
  bool m_is_copy_host = false;
  bool m_is_copy_device = false;
  bool m_finish_reduction = false;

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionLocTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       && (sizeof(CudaReductionLocBlockType<T>)
           <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
      "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
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
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>
{
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceMaxLoc(T init_val, Index_type init_loc)
  {
    m_is_copy_host = false;
    m_myID = getCudaReductionId();
    getCudaReductionMemBlock(m_myID, (void **)&m_blockdata);
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_tally_host,
                               (void **)&m_tally_device);
    m_tally_host->tally.val = init_val;
    m_tally_host->tally.idx = init_loc;
    m_tally_host->retiredBlocks = static_cast<GridSizeType>(0);
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  __host__ __device__
  ReduceMaxLoc(const ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
  {
    *this = other;
#if defined(__CUDA_ARCH__)
    m_is_copy_device = true;
    m_finish_reduction = !other.m_is_copy_device;
    extern __shared__ unsigned char sd_block[];
    T *sd_val = reinterpret_cast<T *>(&sd_block[m_smem_offset]);
    Index_type *sd_idx = reinterpret_cast<Index_type *>(
        &sd_block[m_smem_offset + sizeof(T) * BLOCK_SIZE]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    T val = m_tally_device->tally.val;
    Index_type idx = m_tally_device->tally.idx;
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        sd_val[threadId + i] = val;
        sd_idx[threadId + i] = idx;
      }
    }
    if (threadId < 1) {
      sd_val[threadId] = val;
      sd_idx[threadId] = idx;
    }

    __syncthreads();
#else
    m_is_copy_host = true;
    m_smem_offset =
        getCudaSharedmemOffset(m_myID, BLOCK_SIZE, 
                               (sizeof(T) + sizeof(Index_type)));
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the global memory block chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  __host__ __device__ ~ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_finish_reduction) {
      extern __shared__ unsigned char sd_block[];
      T *sd_val = reinterpret_cast<T *>(&sd_block[m_smem_offset]);
      Index_type *sd_idx = reinterpret_cast<Index_type *>(
          &sd_block[m_smem_offset + sizeof(T) * BLOCK_SIZE]);

      int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;

      int blocks = gridDim.x * gridDim.y * gridDim.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      __syncthreads();

      for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          RAJA_CUDA_MAXLOC(sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId + i],
                           sd_idx[threadId + i]);
        }
        __syncthreads();
      }

      for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
        if (threadId < i) {
          RAJA_CUDA_MAXLOC(sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId],
                           sd_idx[threadId],
                           sd_val[threadId + i],
                           sd_idx[threadId + i]);
        }
      }

      bool lastBlock = false;
      if (threadId < 1) {
        m_blockdata->values[blockId] = sd_val[threadId];
        m_blockdata->indices[blockId] = sd_idx[threadId];

        __threadfence();
        unsigned int oldBlockCount =
            atomicInc((unsigned int *)&m_tally_device->retiredBlocks,
                      (blocks - 1));
        lastBlock = (oldBlockCount == (blocks - 1));
      }
      lastBlock = __syncthreads_or(lastBlock);

      if (lastBlock) {
        CudaReductionLocType<T> lmax{sd_val[0], sd_idx[0]};

        int threads = blockDim.x * blockDim.y * blockDim.z;
        for (int i = threadId; i < blocks; i += threads) {
          RAJA_CUDA_MAXLOC(lmax.val,
                           lmax.idx,
                           lmax.val,
                           lmax.idx,
                           m_blockdata->values[i],
                           m_blockdata->indices[i]);
        }
        sd_val[threadId] = lmax.val;
        sd_idx[threadId] = lmax.idx;
        __syncthreads();

        for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
          if (threadId < i) {
            RAJA_CUDA_MAXLOC(sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId + i],
                             sd_idx[threadId + i]);
          }
          __syncthreads();
        }

        for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
          if (threadId < i) {
            RAJA_CUDA_MAXLOC(sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId],
                             sd_idx[threadId],
                             sd_val[threadId + i],
                             sd_idx[threadId + i]);
          }
        }

        if (threadId < 1) {
          RAJA_CUDA_MAXLOC(m_tally_device->tally.val,
                           m_tally_device->tally.idx,
                           m_tally_device->tally.val,
                           m_tally_device->tally.idx,
                           sd_val[threadId],
                           sd_idx[threadId]);
        }
      }
    }
#else
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif

    
  }

  /*!
   * \brief Operator that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally.val;
  }

  /*!
   * \brief Method that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Method that returns index value corresponding to the reduced max.
   *
   * Note: accessor only executes on host.
   */
  Index_type getLoc()
  {
    if (Async) {
      beforeCudaReadTallyBlockAsync(m_myID);
    } else {
      beforeCudaReadTallyBlock(m_myID);
    }
    return m_tally_host->tally.idx;
  }

  /*!
   * \brief Method that updates max and index values.
   *
   * Note: only operates on device.
   */
  __device__ ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T> const &maxloc(
      T val,
      Index_type idx) const
  {
    extern __shared__ unsigned char sd_block[];
    T *sd_val = reinterpret_cast<T *>(&sd_block[m_smem_offset]);
    Index_type *sd_idx = reinterpret_cast<Index_type *>(
        &sd_block[m_smem_offset + sizeof(T) * BLOCK_SIZE]);

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    RAJA_CUDA_MAXLOC(sd_val[threadId],
                     sd_idx[threadId],
                     sd_val[threadId],
                     sd_idx[threadId],
                     val,
                     idx);
    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>();

  /*!
   * \brief Pointer to host tally block cache slot for this reduction variable.
   */
  CudaReductionLocTallyType<T> *m_tally_host = nullptr;

  /*!
   * \brief Pointer to device data block for this reduction variable.
   */
  CudaReductionLocBlockType<T> *m_blockdata = nullptr;

  /*!
   * \brief Pointer to device tally block slot for this reduction variable.
   */
  CudaReductionLocTallyType<T> *m_tally_device = nullptr;

  /*!
   * \brief My cuda reduction variable ID.
   */
  int m_myID = -1;

  /*!
   * \brief Byte offset into dynamic shared memory.
   */
  int m_smem_offset = -1;

  /*!
   * \brief If this variable is a copy or not; only original may release memory 
   *        or perform finalization.
   */
  bool m_is_copy_host = false;
  bool m_is_copy_device = false;
  bool m_finish_reduction = false;

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= 32) && (BLOCK_SIZE <= 1024));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionLocTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       && (sizeof(CudaReductionLocBlockType<T>)
           <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
      "Error: type must be of size <= " MACROSTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
