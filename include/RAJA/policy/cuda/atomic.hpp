/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations for CUDA
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_atomic_HPP
#define RAJA_policy_cuda_atomic_HPP

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
#include "RAJA/util/defines.hpp"
#include "RAJA/util/Operators.hpp"

#include <stdexcept>

#if defined(RAJA_ENABLE_CUDA)

//
// Determine which atomics we need to implement ourselves with CAS primitive
//

#if __CUDA_ARCH__ < 600

#define RAJA_CUDA_DOUBLE_ATOMIC_ADD_CAS

#endif


#if __CUDA_ARCH__ < 350

#define RAJA_CUDA_UINT64_ATOMIC_MINMAX_CAS

#endif



namespace RAJA
{
struct cuda_atomic{};



/*!
 * Catch-all policy passes off to CUDA's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 */
template<typename T>
RAJA_INLINE
__device__
T atomicAdd(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
	return ::atomicAdd((T*)acc, value);
}

// Before Pascal's, no native support for double-precision atomic add
// So we use the CAS approach
#ifdef RAJA_CUDA_DOUBLE_ATOMIC_ADD_CAS

template<>
RAJA_INLINE
__device__
double atomicAdd<double>(cuda_atomic, double volatile *acc, double value){
  unsigned long long oldval, newval, readback;
  oldval = __double_as_longlong(*acc);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = ::atomicCAS((unsigned long long *)acc, oldval, newval))
         != oldval) {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
  return __longlong_as_double(oldval);
}


#endif // RAJA_CUDA_DOUBLE_ATOMIC_ADD_CAS



template<typename T>
RAJA_INLINE
__device__
T atomicSub(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
  return ::atomicSub((T*)acc, value);
}

template<>
RAJA_INLINE
__device__
float atomicSub<float>(cuda_atomic, float volatile *acc, float value){
	return RAJA::atomicAdd<float>(cuda_atomic{}, acc, -value);
}


template<>
RAJA_INLINE
__device__
double atomicSub<double>(cuda_atomic, double volatile *acc, double value){
	return RAJA::atomicAdd<double>(cuda_atomic{}, acc, -value);
}


template<typename T>
RAJA_INLINE
__device__
T atomicMin(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
  return ::atomicMin((T*)acc, value);
}

template<typename T>
RAJA_INLINE
__device__
T atomicMax(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
  return ::atomicMax((T*)acc, value);
}





template<>
RAJA_INLINE
__device__
float atomicMin<float>(cuda_atomic, float volatile *address, float value){
  float temp = *address;
  if (temp > value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          ::atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MIN(__int_as_float(assumed), value)));
    } while (assumed != oldval);
    temp = __int_as_float(oldval);
  }
  return temp;
}


template<>
RAJA_INLINE
__device__
double atomicMin<double>(cuda_atomic, double volatile *address, double value){
  double temp = *address;
  if (temp > value) {
    unsigned long long *address_as_ull =
        const_cast<unsigned long long *>(
           reinterpret_cast<unsigned long long volatile *>(address)
				);
    unsigned long long assumed;
    unsigned long long oldval = __double_as_longlong(temp);
    do {
      assumed = oldval;
      oldval =
          ::atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(
                      RAJA_MIN(__longlong_as_double(assumed), value)
                    ));
    } while (assumed != oldval);
    temp = __longlong_as_double(oldval);
  }
  return temp;
}

template<>
RAJA_INLINE
__device__
float atomicMax<float>(cuda_atomic, float volatile *address, float value){
  float temp = *address;
  if (temp < value) {
    int *address_as_i = (int *)address;
    int assumed;
    int oldval = __float_as_int(temp);
    do {
      assumed = oldval;
      oldval =
          ::atomicCAS(address_as_i,
                    assumed,
                    __float_as_int(RAJA_MAX(__int_as_float(assumed), value)));
    } while (assumed != oldval);
    temp = __int_as_float(oldval);
  }
  return temp;
}


template<>
RAJA_INLINE
__device__
double atomicMax<double>(cuda_atomic, double volatile *address, double value){
  double temp = *address;
  if (temp < value) {
    unsigned long long *address_as_ull =
        const_cast<unsigned long long *>(
           reinterpret_cast<unsigned long long volatile *>(address)
				);
    unsigned long long assumed;
    unsigned long long oldval = __double_as_longlong(temp);
    do {
      assumed = oldval;
      oldval =
          ::atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(
                      RAJA_MAX(__longlong_as_double(assumed), value)
                    ));
    } while (assumed != oldval);
    temp = __longlong_as_double(oldval);
  }
  return temp;
}


// Before sm_35, no native support for uint64 min and max,
// so we implement a CAS approach
#ifdef RAJA_CUDA_UINT64_ATOMIC_MINMAX_CAS

template<>
RAJA_INLINE
__device__
unsigned long long atomicMin<unsigned long long>(cuda_atomic, unsigned long long volatile *address, unsigned long long value){
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp > value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval = ::atomicCAS(const_cast<unsigned long long *>(address), assumed, RAJA_MIN(assumed, value));
    } while (assumed != oldval);
    temp = oldval;
  }
  return temp;
}

template<>
RAJA_INLINE
__device__
unsigned long long atomicMax<unsigned long long>(cuda_atomic, unsigned long long volatile *address, unsigned long long value){
  unsigned long long int temp =
      *(reinterpret_cast<unsigned long long int volatile *>(address));
  if (temp < value) {
    unsigned long long int assumed;
    unsigned long long int oldval = temp;
    do {
      assumed = oldval;
      oldval = ::atomicCAS(const_cast<unsigned long long *>(address), assumed, RAJA_MAX(assumed, value));
    } while (assumed != oldval);
    temp = oldval;
  }
  return temp;
}


#endif




template<typename T>
RAJA_INLINE
__device__
T atomicInc(cuda_atomic, T volatile *acc){
  // builtin only exists for unsigned, so default to using atomicAdd
	return RAJA::atomicAdd(cuda_atomic{}, (T*)acc, (T)1);
}

template<>
RAJA_INLINE
__device__
unsigned atomicInc<unsigned>(cuda_atomic, unsigned volatile *acc){
  // the CUDA builtin atomic only exists for unsigned ints
  return ::atomicInc((unsigned*)acc, (unsigned)RAJA::operators::limits<unsigned>::max());
}

template<typename T>
RAJA_INLINE
__device__
T atomicDec(cuda_atomic, T volatile *acc){
  // builtin only exists for unsigned, so default to using atomicSub
	return RAJA::atomicSub(cuda_atomic{}, (T*)acc, (T)1);
}

template<>
RAJA_INLINE
__device__
unsigned atomicDec<unsigned>(cuda_atomic, unsigned volatile *acc){
  // the CUDA builtin atomic only exists for unsigned ints
  return ::atomicDec((unsigned*)acc, (unsigned)RAJA::operators::limits<unsigned>::max());
}


template<typename T>
RAJA_INLINE
__device__
T atomicAnd(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
	return ::atomicAnd((T*)acc, value);
}


template<typename T>
RAJA_INLINE
__device__
T atomicOr(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
	return ::atomicOr((T*)acc, value);
}


template<typename T>
RAJA_INLINE
__device__
T atomicXor(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
	return ::atomicXor((T*)acc, value);
}


template<typename T>
RAJA_INLINE
__device__
T atomicExchange(cuda_atomic, T volatile *acc, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
	return ::atomicExch((T*)acc, value);
}


template<typename T>
RAJA_INLINE
__device__
T atomicCAS(cuda_atomic, T volatile *acc, T compare, T value){
  // attempt to use the CUDA builtin atomic, if it exists for T
	return ::atomicCAS((T*)acc, compare, value);
}




}  // namespace RAJA


// Cleanup gaurds so they don't leak
#undef RAJA_CUDA_DOUBLE_ATOMIC_ADD_CAS
#undef RAJA_CUDA_UINT64_ATOMIC_MINMAX_CAS


#endif // RAJA_ENABLE_CUDA
#endif // guard
