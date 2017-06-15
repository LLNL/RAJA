/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA CUDA policy definitions.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_HPP
#define RAJA_policy_cuda_HPP

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
// For additional details, please also read RAJA/LICENSE.
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

#include "RAJA/config.hpp"
#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{

#if defined(RAJA_ENABLE_CLANG_CUDA)
using cuda_dim_t = uint3;
#else
using cuda_dim_t = dim3;
#endif

///
/////////////////////////////////////////////////////////////////////
///
/// Generalizations of CUDA dim3 x, y and z used to describe
/// sizes and indices for threads and blocks.
///
/////////////////////////////////////////////////////////////////////
///
struct Dim3x {
  __host__ __device__ inline unsigned int &operator()(cuda_dim_t &dim)
  {
    return dim.x;
  }

  __host__ __device__ inline unsigned int operator()(cuda_dim_t const &dim)
  {
    return dim.x;
  }
};
///
struct Dim3y {
  __host__ __device__ inline unsigned int &operator()(cuda_dim_t &dim)
  {
    return dim.y;
  }

  __host__ __device__ inline unsigned int operator()(cuda_dim_t const &dim)
  {
    return dim.y;
  }
};
///
struct Dim3z {
  __host__ __device__ inline unsigned int &operator()(cuda_dim_t &dim)
  {
    return dim.z;
  }

  __host__ __device__ inline unsigned int operator()(cuda_dim_t const &dim)
  {
    return dim.z;
  }
};

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

namespace detail
{
template <bool Async>
struct get_launch {
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template <>
struct get_launch<false> {
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}

template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_exec
    : public RAJA::make_policy_launch_pattern<RAJA::Policy::cuda,
                                              detail::get_launch<Async>::value,
                                              RAJA::Pattern::forall> {
};

//
// NOTE: There is no Index set segment iteration policy for CUDA
//

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction reduction policies
///
///////////////////////////////////////////////////////////////////////
///

template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_reduce
    : public RAJA::make_policy_launch_pattern<RAJA::Policy::cuda,
                                              detail::get_launch<Async>::value,
                                              RAJA::Pattern::reduce> {
};

template <size_t BLOCK_SIZE>
using cuda_reduce_async = cuda_reduce<BLOCK_SIZE, true>;

template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_reduce_atomic
    : public RAJA::make_policy_launch_pattern<RAJA::Policy::cuda,
                                              detail::get_launch<Async>::value,
                                              RAJA::Pattern::reduce> {
};

template <size_t BLOCK_SIZE>
using cuda_reduce_atomic_async = cuda_reduce_atomic<BLOCK_SIZE, true>;

//
// Operations in the included files are parametrized using the following
// values for CUDA warp size and max block size.
//
const int WARP_SIZE = 32;
const int RAJA_CUDA_MAX_BLOCK_SIZE = 2048;

/*!
 * \def RAJA_CUDA_LAUNCH_PARAMS(gridSize, blockSize)
 * Macro that generates kernel launch parameters.
 */
#define RAJA_CUDA_LAUNCH_PARAMS(gridSize, blockSize) \
  gridSize, blockSize, getCudaSharedmemAmount(gridSize, blockSize)

//
// Three different variants of min/max reductions can be run by choosing
// one of these macros. Only one should be defined!!!
//
//#define RAJA_USE_ATOMIC_ONE
#define RAJA_USE_ATOMIC_TWO
//#define RAJA_USE_NO_ATOMICS

#define ull_to_double(x) __longlong_as_double(x)
#define double_to_ull(x) __double_as_longlong(x)

/*!
 ******************************************************************************
 *
 * \brief Generics of atomic update methods used in reduction variables.
 *
 * The generic version just wraps the nvidia cuda atomics.
 * Specializations implement other atomics using atomic CAS.
 *
 ******************************************************************************
 */
template <typename T>
__device__ inline T _atomicMin(T *address, T value)
{
  return atomicMin(address, value);
}
///
template <typename T>
__device__ inline T _atomicMax(T *address, T value)
{
  return atomicMax(address, value);
}
///
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
///
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
/*!
 ******************************************************************************
 *
 * \brief Atomic add update methods used to accumulate to memory locations.
 *
 ******************************************************************************
 */
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
/*!
 ******************************************************************************
 *
 * \brief Atomic add update methods used to accumulate to memory locations.
 *
 ******************************************************************************
 */
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

#elif !defined(RAJA_USE_NO_ATOMICS)

#error one of the options for using/not using atomics must be specified

#endif

}  // closing brace for RAJA namespace

#endif
