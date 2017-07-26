/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA CUDA atomic functions.
 *
 ******************************************************************************
 */

#ifndef RAJA_atomic_cuda_HPP
#define RAJA_atomic_cuda_HPP

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

namespace RAJA
{

namespace cuda
{

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
 * \brief Generics of atomic update methods used in reduction variables.
 *
 * The generic version just wraps the nvidia cuda atomics.
 * Specializations implement other atomics using atomic CAS.
 *
 ******************************************************************************
 */
template <typename T>
__device__ inline T atomicMin(T *address, T value)
{
  return ::atomicMin(address, value);
}
///
template <typename T>
__device__ inline T atomicMax(T *address, T value)
{
  return ::atomicMax(address, value);
}
///
template <typename T>
__device__ inline T atomicAdd(T *address, T value)
{
  return ::atomicAdd(address, value);
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
__device__ inline double atomicMin(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp > value) {
    unsigned long long oldval, newval, readback;
    oldval = __double_as_longlong(temp);
    newval = __double_as_longlong(value);
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    while ((readback = ::atomicCAS(address_as_ull, oldval, newval)) != oldval) {
      oldval = readback;
      newval = __double_as_longlong(
          RAJA_MIN(__longlong_as_double(oldval), value));
    }
    temp = __longlong_as_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float atomicMin(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp > value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = ::atomicCAS(address_as_i, oldval, newval)) != oldval) {
      oldval = readback;
      newval = __float_as_int(RAJA_MIN(__int_as_float(oldval), value));
    }
    temp = __int_as_float(oldval);
  }
  return temp;
}
///
template <>
__device__ inline double atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp < value) {
    unsigned long long oldval, newval, readback;
    oldval = __double_as_longlong(temp);
    newval = __double_as_longlong(value);
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    while ((readback = ::atomicCAS(address_as_ull, oldval, newval)) != oldval) {
      oldval = readback;
      newval = __double_as_longlong(
          RAJA_MAX(__longlong_as_double(oldval), value));
    }
    temp = __longlong_as_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float atomicMax(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp < value) {
    int oldval, newval, readback;
    oldval = __float_as_int(temp);
    newval = __float_as_int(value);
    int *address_as_i = reinterpret_cast<int *>(address);

    while ((readback = ::atomicCAS(address_as_i, oldval, newval)) != oldval) {
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
__device__ inline unsigned long long int atomicMin(
    unsigned long long int *address,
    unsigned long long int value)
{
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp > value) {
    unsigned long long int oldval, newval;
    oldval = temp;
    newval = value;

    while ((readback = ::atomicCAS(address, oldval, newval)) != oldval) {
      oldval = readback;
      newval = RAJA_MIN(oldval, value);
    }
  }
  return readback;
}
///
template <>
__device__ inline unsigned long long int atomicMax(
    unsigned long long int *address,
    unsigned long long int value)
{
  unsigned long long int readback =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (readback < value) {
    unsigned long long int oldval, newval;
    oldval = readback;
    newval = value;

    while ((readback = ::atomicCAS(address, oldval, newval)) != oldval) {
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
__device__ inline double atomicAdd(double *address, double value)
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = ::atomicCAS((unsigned long long *)address, oldval, newval))
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
__device__ inline double atomicMin(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp > value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = __double_as_longlong(temp);
    do {
      assumed = oldval;
      oldval = ::atomicCAS(address_as_ull, assumed,
                           __double_as_longlong(
                               RAJA_MIN(__longlong_as_double(assumed), value)));
    } while (assumed != oldval);
    temp = __longlong_as_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float atomicMin(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp > value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval = ::atomicCAS(address_as_i, assumed,
                           __float_as_int(
                               RAJA_MIN(__int_as_float(assumed), value)));
    } while (assumed != oldval);
    temp = __int_as_float(oldval);
  }
  return temp;
}
///
template <>
__device__ inline double atomicMax(double *address, double value)
{
  double temp = *(reinterpret_cast<double volatile *>(address));
  if (temp < value) {
    unsigned long long *address_as_ull =
        reinterpret_cast<unsigned long long *>(address);

    unsigned long long assumed;
    unsigned long long oldval = __double_as_longlong(temp);
    do {
      assumed = oldval;
      oldval = ::atomicCAS(address_as_ull, assumed,
                           __double_as_longlong(
                               RAJA_MAX(__longlong_as_double(assumed), value)));
    } while (assumed != oldval);
    temp = __longlong_as_double(oldval);
  }
  return temp;
}
///
template <>
__device__ inline float atomicMax(float *address, float value)
{
  float temp = *(reinterpret_cast<float volatile *>(address));
  if (temp < value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval = ::atomicCAS(address_as_i, assumed,
                           __float_as_int(
                               RAJA_MAX(__int_as_float(assumed), value)));
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
__device__ inline unsigned long long int atomicMin(
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
      oldval = ::atomicCAS(address, assumed, RAJA_MIN(assumed, value));
    } while (assumed != oldval);
    temp = oldval;
  }
  return temp;
}
///
template <>
__device__ inline unsigned long long int atomicMax(
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
      oldval = ::atomicCAS(address, assumed, RAJA_MAX(assumed, value));
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
__device__ inline double atomicAdd(double *address, double value)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int oldval = *address_as_ull, assumed;

  do {
    assumed = oldval;
    oldval = ::atomicCAS(address_as_ull, assumed,
                         __double_as_longlong(
                             __longlong_as_double(oldval) + value));
  } while (assumed != oldval);
  return __longlong_as_double(oldval);
}
#endif

#elif !defined(RAJA_USE_NO_ATOMICS)

#error one of the options for using/not using atomics must be specified

#endif

} // namespace cuda

}  // closing brace for RAJA namespace

#endif
