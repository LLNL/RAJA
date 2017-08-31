/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining sequential atomic operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_sequential_atomic_HPP
#define RAJA_policy_sequential_atomic_HPP

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

namespace RAJA
{
namespace atomic
{

struct seq_atomic{};


RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicAdd(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc += value;
	return ret;
}


RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicSub(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc -= value;
	return ret;
}


RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicMin(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc = ret < value ? ret : value;
	return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicMax(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc = ret > value ? ret : value;
	return ret;
}


RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicInc(seq_atomic, T volatile *acc){
  T ret = *acc;
	(*acc) += 1;
	return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicInc(seq_atomic, T volatile *acc, T val){
  T old = *acc;
	(*acc) = ((old >= val) ? 0 : (old+1));
	return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicDec(seq_atomic, T volatile *acc){
  T ret = *acc;
  (*acc) -= 1;
	return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicDec(seq_atomic, T volatile *acc, T val){
  T old = *acc;
  (*acc) = (((old==0)|(old>val))?val:(old-1));
	return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicAnd(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc &= value;
	return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicOr(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc |= value;
	return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicXor(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc ^= value;
	return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicExchange(seq_atomic, T volatile *acc, T value){
  T ret = *acc;
  *acc = value;
	return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
T atomicCAS(seq_atomic, T volatile *acc, T compare, T value){
  T ret = *acc;
	*acc = ret == compare ? value : ret;
	return ret;
}


}  // namespace atomic
}  // namespace RAJA


#endif // guard
